#include "runner/job.h"

#include <android/log.h>

#include <algorithm>
#include <map>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>

#include <fcntl.h>      // open（埋め込み表の mmap。EdgeSLM_AndroidScale）
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "runner/progress.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "EdgeSLMRunner", __VA_ARGS__)

namespace edgeslm {

using nlohmann::json;

namespace {

// 入力バッファを用意する（v0.2 で多入力対応）。
//
// 型ごとに作り方を変える。**整数入力を乱数バイトで埋めてはいけない**:
// int64 を乱数バイトで埋めると天文学的な値になり、Gather の索引なら範囲外参照、
// position_ids なら意味のない位置になる。実際の入力分布に寄せる。
bool build_inputs(const Session& s, const json& spec,
                  std::vector<std::vector<uint8_t>>* out, std::string* err) {
    const json d = s.describe();
    if (!d.contains("inputs") || d["inputs"].empty()) {
        *err = "model has no inputs";
        return false;
    }

    const std::string kind = spec.value("kind", std::string("random"));
    if (kind != "zeros" && kind != "random") {
        *err = "unsupported input.kind: " + kind + " (supports zeros/random)";
        return false;
    }
    // 名前ごとの上限（Gather の索引など）。外すと範囲外参照になる。
    const json int_max = spec.value("int_max", json::object());

    out->clear();
    std::mt19937 rng(spec.value("seed", 0u));

    for (const auto& in : d["inputs"]) {
        if (!in.value("static", false)) {
            // QNN EP は動的形状を扱えない（要件定義書 3.3.2節#11）。
            *err = "input '" + in.value("name", std::string("?")) +
                   "' shape is not static; QNN EP requires fixed shapes";
            return false;
        }
        size_t count = 1;
        for (const auto& dim : in["shape"]) count *= static_cast<size_t>(dim.get<int64_t>());
        const std::string type = in.value("type", std::string("float32"));
        const std::string name = in.value("name", std::string());

        size_t elem = 4;
        if (type == "uint8" || type == "int8") elem = 1;
        else if (type == "uint16" || type == "int16" || type == "float16") elem = 2;
        else if (type == "int64") elem = 8;

        std::vector<uint8_t> buf(count * elem, 0);
        if (kind == "random") {
            if (type == "float32") {
                auto* p = reinterpret_cast<float*>(buf.data());
                // 活性化は 0..1 に収まらない。負側も踏ませる
                // （ホスト側のキャリブレーションと同じ値域にしないと分布がずれる）。
                std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
                for (size_t i = 0; i < count; ++i) p[i] = dist(rng);
            } else if (type == "int64") {
                auto* p = reinterpret_cast<int64_t*>(buf.data());
                const int64_t hi = int_max.contains(name)
                                 ? int_max[name].get<int64_t>()
                                 : static_cast<int64_t>(count) - 1;
                for (size_t i = 0; i < count; ++i) {
                    p[i] = (hi <= 0) ? 0 : static_cast<int64_t>(i % (hi + 1));
                }
            } else if (type == "int32") {
                auto* p = reinterpret_cast<int32_t*>(buf.data());
                const int32_t hi = int_max.contains(name)
                                 ? int_max[name].get<int32_t>()
                                 : static_cast<int32_t>(count) - 1;
                for (size_t i = 0; i < count; ++i) {
                    p[i] = (hi <= 0) ? 0 : static_cast<int32_t>(i % (hi + 1));
                }
            } else {
                std::uniform_int_distribution<int> dist(0, 255);
                for (size_t i = 0; i < buf.size(); ++i) buf[i] = static_cast<uint8_t>(dist(rng));
            }
        }
        out->push_back(std::move(buf));
    }
    return true;
}

// 単入力版（bench 用）。多入力モデルはここで明示的に弾く。
bool build_input(const Session& s, const json& spec, std::vector<uint8_t>* out,
                 std::string* err) {
    std::vector<std::vector<uint8_t>> all;
    if (!build_inputs(s, spec, &all, err)) return false;
    if (all.size() != 1) {
        *err = "bench supports single-input models only; this model has " +
               std::to_string(all.size()) + " inputs (use task=op_check)";
        return false;
    }
    *out = std::move(all[0]);
    return true;
}

// 生のバイト列としてファイルを読む。**サイズが期待と違えば失敗させる**——
// 黙って足りないまま推論すると、結果が静かに壊れる。
bool read_all(const std::string& path, std::vector<uint8_t>* out, std::string* err) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { *err = "cannot open: " + path; return false; }
    const std::streamsize n = f.tellg();
    f.seekg(0);
    out->resize(static_cast<size_t>(n));
    if (n > 0 && !f.read(reinterpret_cast<char*>(out->data()), n)) {
        *err = "read failed: " + path;
        return false;
    }
    return true;
}

bool write_all(const std::string& path, const std::vector<uint8_t>& data, std::string* err) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) { *err = "cannot write: " + path; return false; }
    if (!data.empty()) f.write(reinterpret_cast<const char*>(data.data()),
                               static_cast<std::streamsize>(data.size()));
    if (!f) { *err = "write failed: " + path; return false; }
    return true;
}

// fp16 → fp32（ビット操作。ツールチェーンの __fp16 対応に依存しない）
inline float half_to_float(uint16_t h) {
    const uint32_t s = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu, f;
    if (e == 0) {
        if (m == 0) {
            f = s;
        } else {                                   // 非正規化数
            int ee = 1;
            while (!(m & 0x400u)) { m <<= 1; --ee; }
            m &= 0x3ffu;
            f = s | (static_cast<uint32_t>(ee + 127 - 15) << 23) | (m << 13);
        }
    } else if (e == 31) {
        f = s | 0x7f800000u | (m << 13);
    } else {
        f = s | ((e + 127 - 15) << 23) | (m << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

// 埋め込み表（fp16 の [V][H]）を読み取り専用で mmap する。
// **全体をヒープに読まない**のは、1.5B は NPU 側で 1.5GB を使い端末メモリに余裕が無いため
// （EdgeSLM_AndroidScale P0A/P2）。触った行のページだけがページキャッシュに載る。
struct MappedFile {
    const uint8_t* p = nullptr;
    size_t n = 0;
    bool open(const std::string& path, std::string* err) {
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) { *err = "cannot open: " + path; return false; }
        struct stat st {};
        if (fstat(fd, &st) != 0 || st.st_size <= 0) {
            ::close(fd);
            *err = "cannot stat: " + path;
            return false;
        }
        void* m = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (m == MAP_FAILED) { *err = "mmap failed: " + path; return false; }
        p = static_cast<const uint8_t*>(m);
        n = static_cast<size_t>(st.st_size);
        return true;
    }
    ~MappedFile() { if (p) munmap(const_cast<uint8_t*>(p), n); }
};

double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

json stats_of(std::vector<double> v) {
    json j;
    if (v.empty()) return j;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    auto pct = [&](double p) {
        const double idx = p * (static_cast<double>(n) - 1.0);
        const size_t lo = static_cast<size_t>(std::floor(idx));
        const size_t hi = static_cast<size_t>(std::ceil(idx));
        if (lo == hi) return v[lo];
        return v[lo] + (v[hi] - v[lo]) * (idx - static_cast<double>(lo));
    };
    double sum = 0.0;
    for (double x : v) sum += x;
    const double mean = sum / static_cast<double>(n);
    double var = 0.0;
    for (double x : v) var += (x - mean) * (x - mean);

    // 平均は主指標にしない（基本設計① 5章）。初回の外れ値に引きずられるため。
    j["median"] = pct(0.5);
    j["p90"]    = pct(0.9);
    j["min"]    = v.front();
    j["max"]    = v.back();
    j["mean"]   = mean;
    j["stddev"] = (n > 1) ? std::sqrt(var / static_cast<double>(n - 1)) : 0.0;
    j["n"]      = n;
    return j;
}

}  // namespace

bool is_supported_task(const std::string& task) {
    return task == "probe" || task == "bench" || task == "op_check" || task == "infer" ||
           task == "generate" || task == "generate_kv";
}

json run_job(Session& session, const json& job, const bool* cancel_flag) {
    json out;
    const std::string task = job.value("task", std::string("probe"));
    out["job_id"] = job.value("job_id", std::string(""));
    out["task"] = task;
    out["describe"] = session.describe();
    out["versions"] = Session::versions();
    // 電力プロファイルは制御できない。全レコードに固定で刻み、
    // 他者の burst 測定値や公称値と混ぜられないようにする（基本設計① 2.2節 / 6章）。
    out["power_profile"] = "default(unsigned_pd)";

    if (!is_supported_task(task)) {
        out["ok"] = false;
        out["error"] = "task '" + task +
                       "' is not implemented (probe/bench/op_check/infer/generate/generate_kv)";
        return out;
    }

    if (task == "probe") {
        out["ok"] = session.ok();
        return out;
    }

    if (task == "infer") {
        // 実データを読んで推論し、出力を生バイト列で書き出す。
        // ホスト側が fp32 の参照と突き合わせて**実機での精度**を出す。
        const json batches = job.value("batches", json::array());
        const std::string out_dir = job.value("output", json::object())
                                       .value("dir", std::string("out"));
        const json desc = session.describe();
        json records = json::array();
        int n_ok = 0;

        for (const auto& b : batches) {
            json rec;
            const std::string id = b.value("id", std::string("0"));
            rec["id"] = id;
            if (cancel_flag && *cancel_flag) { rec["error"] = "cancelled"; records.push_back(rec); break; }

            // 入力を名前順（モデルの入力順）に読む
            std::vector<std::vector<uint8_t>> inputs;
            std::string err;
            bool loaded = true;
            const json files = b.value("inputs", json::object());
            for (const auto& in : desc["inputs"]) {
                const std::string name = in.value("name", std::string());
                if (!files.contains(name)) {
                    err = "no input file for '" + name + "'";
                    loaded = false;
                    break;
                }
                std::vector<uint8_t> buf;
                if (!read_all(session.resolve_path(files[name].get<std::string>()), &buf, &err)) {
                    loaded = false;
                    break;
                }
                // 期待バイト数と一致しなければ止める
                size_t count = 1;
                for (const auto& d : in["shape"]) count *= static_cast<size_t>(d.get<int64_t>());
                const std::string type = in.value("type", std::string("float32"));
                size_t esz = 4;
                if (type == "uint8" || type == "int8") esz = 1;
                else if (type == "uint16" || type == "int16" || type == "float16") esz = 2;
                else if (type == "int64") esz = 8;
                if (buf.size() != count * esz) {
                    err = "size mismatch for '" + name + "': expected " +
                          std::to_string(count * esz) + " bytes, got " + std::to_string(buf.size());
                    loaded = false;
                    break;
                }
                inputs.push_back(std::move(buf));
            }
            if (!loaded) {
                rec["error"] = err;
                records.push_back(rec);
                continue;
            }

            std::vector<std::vector<uint8_t>> outs;
            double ms = 0.0;
            if (!session.run(inputs, &ms, &outs)) {
                rec["error"] = session.last_error();
                records.push_back(rec);
                continue;
            }
            rec["ort_run_ms"] = ms;

            json written = json::array();
            bool wrote = true;
            for (size_t i = 0; i < outs.size() && i < desc["outputs"].size(); ++i) {
                const std::string oname = desc["outputs"][i].value("name", std::string("out"));
                const std::string rel = out_dir + "/" + job.value("job_id", std::string("infer")) +
                                        "_" + id + "_" + oname + ".bin";
                if (!write_all(session.resolve_path(rel), outs[i], &err)) {
                    rec["error"] = err;
                    wrote = false;
                    break;
                }
                written.push_back({{"name", oname}, {"path", rel},
                                   {"bytes", outs[i].size()},
                                   {"shape", desc["outputs"][i]["shape"]},
                                   {"type", desc["outputs"][i]["type"]}});
            }
            if (!wrote) { records.push_back(rec); continue; }
            rec["outputs"] = written;
            records.push_back(rec);
            ++n_ok;
            LOGI("infer[%s] %.2f ms, %zu outputs", id.c_str(), ms, outs.size());
        }

        out["records"] = records;
        out["ok"] = (n_ok == static_cast<int>(batches.size())) && n_ok > 0;
        out["completed"] = n_ok;
        return out;
    }

    if (task == "generate") {
        // 窓ずらし生成（KVキャッシュなし）。
        //
        // v68 では seq=1 の `rms_norm` 融合 op が作れず KVキャッシュ decode を組めない
        // （検証結果まとめ 7.2）。そこで**既に実機で動いている prefill モデル（seq=W）**を
        // そのまま使い、窓を1つずつずらして次トークンを取る。
        //
        // 窓の作り方と argmax の位置は EdgeSLM_Android/app/pipeline/generate_on_device.py
        // と**同一の規則**にしてある。あちらは fp32 と 6/6 一致まで確認済みの経路なので、
        // 同じ規則にしておけば両者を突き合わせて「この実装が正しい」と言える。
        //   w = seq[-W:] を右詰め（不足分は左を pad で埋める）→ 見るのは常に位置 W-1
        //
        // ホスト駆動版との違いは**呼び出し方だけ**である:
        //   あちら … 1トークンごとに am start + 1秒ポーリング（体感 約1秒/token）
        //   ここ   … セッションを開いたまま session.run() を回す（ロードは1回きり）
        const json d = session.describe();
        if (!d.contains("inputs") || !d.contains("outputs")) {
            out["ok"] = false;
            out["error"] = "generate: session has no I/O info";
            return out;
        }

        // ---- 入出力の形を検証する。合わなければ黙って走らせない ----
        int ids_idx = -1;
        int64_t W = 0;
        for (size_t i = 0; i < d["inputs"].size(); ++i) {
            const auto& in = d["inputs"][i];
            if (in.value("name", std::string()) != "input_ids") continue;
            ids_idx = static_cast<int>(i);
            if (in.value("type", std::string()) != "int64" || in["shape"].size() != 2 ||
                in["shape"][0].get<int64_t>() != 1) {
                out["ok"] = false;
                out["error"] = "generate: input_ids must be int64 [1,W]";
                return out;
            }
            W = in["shape"][1].get<int64_t>();
        }
        if (ids_idx < 0 || W <= 0) {
            out["ok"] = false;
            out["error"] = "generate: model has no 'input_ids' input";
            return out;
        }

        int logits_idx = -1;
        int64_t V = 0;
        int64_t L = 0;   // logits の位置数。W（全位置）か 1（最終位置のみ）
        for (size_t i = 0; i < d["outputs"].size(); ++i) {
            const auto& o = d["outputs"][i];
            if (o.value("name", std::string()) != "logits") continue;
            logits_idx = static_cast<int>(i);
            L = (o["shape"].size() == 3) ? o["shape"][1].get<int64_t>() : 0;
            // L == 1 を許す（lm_head を最終位置だけに掛けたモデル）。
            // 窓を広げると logits [1,W,V] は無視できない大きさになる:
            //   W=16 -> 9.7MB / W=64 -> 38.9MB（毎ステップ取り出してコピーする）
            // 生成では最終位置しか使わないので、モデル側で切る方が速い。
            // その場合の出力は [1,1,V] = 0.6MB になる。
            if (o.value("type", std::string()) != "float32" || o["shape"].size() != 3 ||
                (L != W && L != 1)) {
                out["ok"] = false;
                out["error"] = "generate: logits must be float32 [1,W,V] or [1,1,V]";
                return out;
            }
            V = o["shape"][2].get<int64_t>();
        }
        if (logits_idx < 0 || V <= 0) {
            out["ok"] = false;
            out["error"] = "generate: model has no 'logits' output";
            return out;
        }

        std::vector<int64_t> prompt;
        for (const auto& v : job.value("prompt_ids", json::array())) {
            prompt.push_back(v.get<int64_t>());
        }
        if (prompt.empty()) {
            out["ok"] = false;
            out["error"] = "generate: prompt_ids is empty";
            return out;
        }

        const int max_new = job.value("max_new", 16);
        const int64_t pad_id = job.value("pad_id", static_cast<int64_t>(151643));
        // attn_mask を毎ステップ作り直すか。
        //
        // 窓を広げると**パディングを塞ぐかどうかが効いてくる**。fp32 で実測した
        // （正解はパディング無しの答え、10プロンプト。2026-09-13）:
        //     W=16 : 正しいマスク 10/10 / 全部見える 6/10
        //     W=64 : 正しいマスク 10/10 / 全部見える 3/10
        // 窓 64 でプロンプトが6トークンなら pad が58個入るので、塞がないと壊れる。
        //
        // 生成が進むと有効長が増える（プロンプト長 -> W で飽和）ので、
        // **固定ファイルでは足りず毎ステップ作る必要がある**。
        const bool dyn_mask = job.value("dynamic_mask", false);
        const std::string mask_name = job.value("mask_input", std::string("attn_mask"));
        // マスクの遮断値。-3.4e38 のような極値は量子化の分解能を食い潰すので使わない
        // （C-9。要件定義書 3.4節）。-60 なら softmax 後は e^-60 = 8.8e-27 で十分。
        const float mask_neg = job.value("mask_neg", -60.0f);
        int mask_idx = -1;
        std::vector<int64_t> stop_ids;
        for (const auto& v : job.value("stop_ids", json::array({151645, 151643}))) {
            stop_ids.push_back(v.get<int64_t>());
        }

        // ---- 入力バッファを1回だけ用意する ----
        // input_ids 以外（マスク入力版の cos/sin など）は生成中に変わらないので
        // ファイルから一度読んで使い回す。ここを用意しておくことで、
        // モデルを差し替えるときに C++ を触らずに済む（job.json だけで済む）。
        std::vector<std::vector<uint8_t>> inputs(d["inputs"].size());
        const json const_inputs = job.value("const_inputs", json::object());
        for (size_t i = 0; i < d["inputs"].size(); ++i) {
            const auto& in = d["inputs"][i];
            const std::string nm = in.value("name", std::string());
            size_t count = 1;
            for (const auto& dim : in["shape"]) count *= static_cast<size_t>(dim.get<int64_t>());
            const std::string type = in.value("type", std::string("float32"));
            size_t esz = 4;
            if (type == "uint8" || type == "int8") esz = 1;
            else if (type == "uint16" || type == "int16" || type == "float16") esz = 2;
            else if (type == "int64") esz = 8;

            if (static_cast<int>(i) == ids_idx) {
                inputs[i].assign(count * esz, 0);
                continue;
            }
            if (dyn_mask && nm == mask_name) {
                // 形と型を確かめてからバッファだけ確保する（中身は毎ステップ書く）
                if (type != "float32" || in["shape"].size() != 4 ||
                    in["shape"][2].get<int64_t>() != W ||
                    in["shape"][3].get<int64_t>() != W) {
                    out["ok"] = false;
                    out["error"] = "generate: dynamic_mask needs float32 '" + nm +
                                   "' of shape [1,1,W,W]";
                    return out;
                }
                mask_idx = static_cast<int>(i);
                inputs[i].assign(count * esz, 0);
                continue;
            }
            if (!const_inputs.contains(nm)) {
                out["ok"] = false;
                out["error"] = "generate: no const_inputs entry for input '" + nm + "'";
                return out;
            }
            std::string err2;
            if (!read_all(session.resolve_path(const_inputs[nm].get<std::string>()),
                          &inputs[i], &err2)) {
                out["ok"] = false;
                out["error"] = "generate: " + err2;
                return out;
            }
            if (inputs[i].size() != count * esz) {
                out["ok"] = false;
                out["error"] = "generate: size mismatch for '" + nm + "': expected " +
                               std::to_string(count * esz) + " bytes, got " +
                               std::to_string(inputs[i].size());
                return out;
            }
        }

        if (dyn_mask && mask_idx < 0) {
            out["ok"] = false;
            out["error"] = "generate: dynamic_mask is on but the model has no '" +
                           mask_name + "' input";
            return out;
        }

        GenProgress& prog = GenProgress::instance();
        prog.begin(max_new, static_cast<int>(W), prompt.size());

        std::vector<int64_t> seq = prompt;      // プロンプト＋生成済み
        std::vector<int64_t> newly;             // 生成した分だけ
        std::vector<double> step_ms, ort_ms;
        std::vector<std::vector<uint8_t>> outs;
        std::string gerr;
        bool stopped_by_eos = false;

        const double t_start = now_ms();
        for (int step = 0; step < max_new; ++step) {
            if (cancel_flag && *cancel_flag) { gerr = "cancelled"; break; }
            const double s0 = now_ms();

            // 窓を作る（右詰め。不足分は左を pad で埋める）
            auto* w = reinterpret_cast<int64_t*>(inputs[ids_idx].data());
            const size_t n = seq.size();
            const size_t take = std::min<size_t>(n, static_cast<size_t>(W));
            const size_t left = static_cast<size_t>(W) - take;
            for (size_t i = 0; i < left; ++i) w[i] = pad_id;
            for (size_t i = 0; i < take; ++i) w[left + i] = seq[n - take + i];

            // 因果マスク + パディングマスクを作る。
            // 規則は EdgeSLM_Android/app/pipeline/export_qwen_decode.py の build_mask()
            // と**同一**（右詰め前提）。ずれると静かに別のモデルを動かすことになる。
            //   有効行 i (i >= off) は off..i が見える
            //   パディング行 i (i < off) は**自分だけ**開ける（softmax が NaN にならないため）
            if (mask_idx >= 0) {
                auto* mk = reinterpret_cast<float*>(inputs[mask_idx].data());
                const int64_t off = static_cast<int64_t>(left);
                for (int64_t i = 0; i < W; ++i) {
                    float* mrow = mk + i * W;
                    for (int64_t j = 0; j < W; ++j) mrow[j] = mask_neg;
                    if (i >= off) {
                        for (int64_t j = off; j <= i; ++j) mrow[j] = 0.0f;
                    } else {
                        mrow[i] = 0.0f;
                    }
                }
            }

            double ms = 0.0;
            if (!session.run(inputs, &ms, &outs)) { gerr = session.last_error(); break; }
            if (outs.size() <= static_cast<size_t>(logits_idx) ||
                outs[logits_idx].size() != static_cast<size_t>(L) * static_cast<size_t>(V) * 4) {
                gerr = "unexpected logits size";
                break;
            }

            // 見るのは**常に最終位置**。窓は右詰めなので、そこが直近のトークンである。
            // モデルが最終位置だけを返す場合（L == 1）は先頭行がそれである。
            const auto* row = reinterpret_cast<const float*>(outs[logits_idx].data()) +
                              static_cast<size_t>(L - 1) * static_cast<size_t>(V);
            int64_t best = 0;
            float bestv = row[0];
            // 同点は**先に出た方**を採る（numpy.argmax と同じ規則。ここを > にしないと
            // ホスト側と結果がずれうる）
            for (int64_t k = 1; k < V; ++k) {
                if (row[k] > bestv) { bestv = row[k]; best = k; }
            }

            const double s1 = now_ms();
            seq.push_back(best);
            newly.push_back(best);
            step_ms.push_back(s1 - s0);
            ort_ms.push_back(ms);
            prog.push(best, s1 - s0, ms);
            LOGI("generate[%d] id=%lld ort=%.2fms step=%.2fms", step,
                 static_cast<long long>(best), ms, s1 - s0);

            if (std::find(stop_ids.begin(), stop_ids.end(), best) != stop_ids.end()) {
                stopped_by_eos = true;
                break;
            }
        }
        const double total_ms = now_ms() - t_start;
        prog.finish(gerr);

        out["new_ids"] = newly;
        out["step_ms"] = step_ms;              // 生サンプルを残す（基本設計① 5章）
        out["ort_run_ms"] = stats_of(ort_ms);  // OrtRun そのもの
        out["step_ms_stats"] = stats_of(step_ms);  // 1トークンの実時間
        out["total_ms"] = total_ms;
        out["window"] = W;
        out["logits_len"] = L;
        out["dynamic_mask"] = dyn_mask;
        out["prompt_len"] = prompt.size();
        // 窓より長いプロンプトは**直近 W トークンしか見えない**。黙って切らずに明示する。
        out["prompt_truncated"] = prompt.size() > static_cast<size_t>(W);
        out["stopped_by_eos"] = stopped_by_eos;
        if (!gerr.empty()) {
            out["ok"] = false;
            out["error"] = gerr;
        } else {
            out["ok"] = !newly.empty();
        }
        return out;
    }


    if (task == "generate_kv" && job.contains("dialogs")) {
        // ★**独立した対話を1つのセッションで続けて回す**（`dialogs` に `turns` の配列。評価用）。
        //
        //   RunnerService はジョブごとにモデルを読み直すので、評価の問題を1問1ジョブにすると
        //   1.5B（1,481MiB）のロードを問題の数だけ繰り返すことになる（1回 3〜5秒、裏の経路なので
        //   lmkd に kill される機会も増える。EdgeSLM_AndroidScale P3）。
        //   対話ごとに**キャッシュを捨ててから**、`turns` と同じ手順で回す。
        json results = json::array();
        json sub = job;
        sub.erase("dialogs");
        for (const auto& dlg : job["dialogs"]) {
            session.kv_carry().reset();
            sub["turns"] = dlg;
            results.push_back(run_job(session, sub, cancel_flag));
            if (cancel_flag && *cancel_flag) break;
        }
        out["ok"] = true;
        out["dialogs"] = results;
        return out;
    }

    if (task == "generate_kv" && job.contains("turns")) {
        // ★**同じセッションで複数ターンを続けて回す**（`turns` に prompt_ids の配列）。
        //
        //   `reuse_cache` の効果は「セッションを跨いで測る」必要があるが、
        //   `RunnerService` はジョブごとにセッションを作り直すので、
        //   ジョブ1本では測れない。ここで内側から回す。
        //   デモ用の対話でもそのまま使える。
        // ★`turns` の各要素は「**前のターンの続きに足すトークン**」である。
        //   全履歴を毎回渡す形にすると、**呼び出し側がモデルの実際の応答を
        //   知らないと履歴を組み立てられない**（応答を推測して組むと前方一致せず、
        //   安全側に倒れてキャッシュが捨てられる。2026-09-14 に実際にやった）。
        //   実際のチャットでも、足すのは「直前の応答を閉じる特殊トークンと、
        //   次のユーザー発話を包むテンプレート」だけで、**応答の中身に依存しない**。
        json turns = json::array();
        json sub = job;
        sub.erase("turns");
        bool first = true;
        for (const auto& delta : job["turns"]) {
            std::vector<int64_t> ids;
            if (!first) {
                const auto& prev = session.kv_carry().tokens;
                ids.assign(prev.begin(), prev.end());     // これまでの実際の履歴
            }
            for (const auto& v : delta) ids.push_back(v.get<int64_t>());
            sub["prompt_ids"] = ids;
            sub["reuse_cache"] = job.value("reuse_cache", true);
            sub["reset_cache"] = first && !job.value("reuse_cache", true);
            first = false;
            json r = run_job(session, sub, cancel_flag);
            r["prompt_len_total"] = static_cast<int64_t>(ids.size());
            turns.push_back(r);
            if (!r.value("ok", false)) break;
        }
        out["ok"] = true;
        out["turns"] = turns;
        return out;
    }

    if (task == "generate_kv") {
        // KVキャッシュを使う decode ループ（seq=1）。
        //
        // ★`generate`（窓ずらし）とは**別の task にしてある**。
        //   窓ずらしは既に実機で検証済みの経路なので、そこへ手を入れずに並べる。
        //   モデルも別（`qwen_dec24_*`）で、同時に開くことはない。
        //
        // 窓ずらしとの違い:
        //   窓ずらし … 毎回 W トークンを入れ直す。過去の計算を毎回やり直す
        //   ここ     … 1トークンだけ入れ、**過去は KVキャッシュで持ち回す**
        //
        // 実測（24層）: 窓16 が 36.7ms/token、窓64 が 42.0ms、**decode は 28.5ms**。
        //
        // キャッシュは「左シフトして末尾に書く」形（長さ L 固定）。
        // マスクは**末尾 n_valid 個だけ開ける**。cos/sin は位置ごとの表から取り出す。
        const json d = session.describe();
        if (!d.contains("inputs") || !d.contains("outputs")) {
            out["ok"] = false;
            out["error"] = "generate_kv: session has no I/O info";
            return out;
        }

        // ---- 入出力の対応づけ ----
        int ids_idx = -1, mask_idx = -1, cos_idx = -1, sin_idx = -1;
        int emb_idx = -1;          // inputs_embeds（埋め込み表を CPU 側に出したモデル。1.5B）
        int64_t L = 0, HD = 0, H = 0;
        std::vector<std::pair<int, std::string>> cache_in;   // (入力番号, past_X の X)
        for (size_t i = 0; i < d["inputs"].size(); ++i) {
            const auto& in = d["inputs"][i];
            const std::string nm = in.value("name", std::string());
            if (nm == "input_ids") {
                ids_idx = static_cast<int>(i);
            } else if (nm == "inputs_embeds") {
                emb_idx = static_cast<int>(i);
                if (in["shape"].size() == 3) H = in["shape"][2].get<int64_t>();
            } else if (nm == "attn_mask") {
                mask_idx = static_cast<int>(i);
                if (in["shape"].size() == 4) L = in["shape"][3].get<int64_t>();
            } else if (nm == "cos") {
                cos_idx = static_cast<int>(i);
                if (in["shape"].size() == 4) HD = in["shape"][3].get<int64_t>();
            } else if (nm == "sin") {
                sin_idx = static_cast<int>(i);
            } else if (nm.rfind("past_", 0) == 0) {
                cache_in.emplace_back(static_cast<int>(i), nm.substr(5));
            }
        }
        if ((ids_idx < 0 && emb_idx < 0) || mask_idx < 0 || cos_idx < 0 || sin_idx < 0 ||
            cache_in.empty()) {
            out["ok"] = false;
            out["error"] = "generate_kv: model needs input_ids (or inputs_embeds) / cos / sin / attn_mask / past_*";
            return out;
        }
        // ---- 埋め込み表（inputs_embeds のモデルだけ）----
        MappedFile emb_tab;
        int64_t emb_rows = 0;
        // 表の型。既定は bf16（Qwen2.5 の重みは bf16 なので、PC 側とビット単位で同じ埋め込みになる）
        const bool emb_bf16 = job.value("embed_dtype", std::string("bf16")) == "bf16";
        if (emb_idx >= 0) {
            std::string e;
            if (!job.contains("embed_table") || H <= 0 ||
                !emb_tab.open(session.resolve_path(job["embed_table"].get<std::string>()), &e)) {
                out["ok"] = false;
                out["error"] = "generate_kv: inputs_embeds model needs embed_table (bf16/fp16 [V][H]) " + e;
                return out;
            }
            emb_rows = static_cast<int64_t>(emb_tab.n / (static_cast<size_t>(H) * 2));
            if (emb_tab.n != static_cast<size_t>(emb_rows) * static_cast<size_t>(H) * 2) {
                out["ok"] = false;
                out["error"] = "generate_kv: embed_table size is not a multiple of H*2";
                return out;
            }
        }

        int logits_idx = -1;
        int64_t V = 0;
        std::map<std::string, int> present_idx;   // X -> 出力番号
        for (size_t i = 0; i < d["outputs"].size(); ++i) {
            const auto& o = d["outputs"][i];
            const std::string nm = o.value("name", std::string());
            if (nm == "logits") {
                logits_idx = static_cast<int>(i);
                if (o["shape"].size() == 3) V = o["shape"][2].get<int64_t>();
            } else if (nm.rfind("present_", 0) == 0) {
                present_idx[nm.substr(8)] = static_cast<int>(i);
            }
        }
        if (logits_idx < 0 || V <= 0) {
            out["ok"] = false;
            out["error"] = "generate_kv: model has no 'logits' output";
            return out;
        }
        for (const auto& c : cache_in) {
            if (!present_idx.count(c.second)) {
                out["ok"] = false;
                out["error"] = "generate_kv: no matching output for past_" + c.second;
                return out;
            }
        }

        std::vector<int64_t> prompt;
        for (const auto& v : job.value("prompt_ids", json::array())) {
            prompt.push_back(v.get<int64_t>());
        }
        if (prompt.empty()) {
            out["ok"] = false;
            out["error"] = "generate_kv: prompt_ids is empty";
            return out;
        }
        const int max_new = job.value("max_new", 16);
        const float mask_neg = job.value("mask_neg", -60.0f);
        std::vector<int64_t> stop_ids;
        for (const auto& v : job.value("stop_ids", json::array({151645, 151643}))) {
            stop_ids.push_back(v.get<int64_t>());
        }

        // ---- cos/sin の表を読む（L×HD の float32。位置ごとに1行使う）----
        const json ci = job.value("const_inputs", json::object());
        std::vector<uint8_t> cos_tab, sin_tab;
        std::string err2;
        for (const char* nm : {"cos_table", "sin_table"}) {
            if (!ci.contains(nm)) {
                out["ok"] = false;
                out["error"] = std::string("generate_kv: const_inputs needs '") + nm +
                               "' (float32 [L][HD])";
                return out;
            }
        }
        if (!read_all(session.resolve_path(ci["cos_table"].get<std::string>()), &cos_tab, &err2) ||
            !read_all(session.resolve_path(ci["sin_table"].get<std::string>()), &sin_tab, &err2)) {
            out["ok"] = false;
            out["error"] = "generate_kv: " + err2;
            return out;
        }
        const size_t row_bytes = static_cast<size_t>(HD) * 4;
        // ★RoPE テーブルの行数は **KVキャッシュの容量とは別物**である。
        //   容量 L は「何トークンぶん覚えるか」、テーブル行数は「位置を何番まで表せるか」。
        //   窓が一杯になっても位置は増え続けるので、**生成したいトークン数ぶん**必要になる。
        const int64_t rope_rows = static_cast<int64_t>(
            std::min(cos_tab.size(), sin_tab.size()) / row_bytes);
        if (rope_rows < L) {
            out["ok"] = false;
            out["error"] = "generate_kv: cos/sin table too small (need at least L*HD float32, got "
                           + std::to_string(rope_rows) + " rows)";
            return out;
        }

        // ---- バッファを1回だけ用意する ----
        std::vector<std::vector<uint8_t>> inputs(d["inputs"].size());
        for (size_t i = 0; i < d["inputs"].size(); ++i) {
            const auto& in = d["inputs"][i];
            size_t count = 1;
            for (const auto& dim : in["shape"]) count *= static_cast<size_t>(dim.get<int64_t>());
            const std::string type = in.value("type", std::string("float32"));
            size_t esz = (type == "int64") ? 8 : 4;
            // キャッシュは**0で初期化する**。マスクで塞ぐので中身は効かないが、
            // 未初期化のまま渡すと NaN が混ざりうる。
            inputs[i].assign(count * esz, 0);
        }

        // ---- 前ターンのキャッシュを引き継ぐ（`reuse_cache`）----
        //
        // 既定は false（従来どおり毎回ゼロから）。true のときだけ、
        // **プロンプトが前回の続きになっているか**を確かめてから再利用する。
        KvCarry& carry = session.kv_carry();
        const bool want_reuse = job.value("reuse_cache", false);
        if (job.value("reset_cache", false)) carry.reset();

        // ★`prompt_is_delta`: `prompt_ids` が「**前のターンの続きに足すぶん**」のとき、
        //   ここで**これまでの履歴を前に付ける**。
        //
        //   これが無いと、呼び出し側が差分だけ送っても
        //   「履歴より短いプロンプト」になって前方一致せず、再利用が効かないまま
        //   **文脈ゼロで生成してしまう**（2026-09-14: UI が毎回
        //   「こんにちは、何かお手伝いできることが…」と会話の冒頭のような返事をしていた）。
        //   差分方式は `turns` ブランチにしか実装しておらず、UI が通る経路に無かった。
        if (job.value("prompt_is_delta", false) && carry.valid && !carry.tokens.empty()) {
            std::vector<int64_t> merged = carry.tokens;
            merged.insert(merged.end(), prompt.begin(), prompt.end());
            prompt.swap(merged);
        }
        size_t skip = 0;                       // 充填を飛ばせるトークン数
        if (want_reuse) {
            if (carry.model != d.value("model", std::string())) {
                carry.reset();                 // モデルが張り替わったら使えない
            }
            skip = carry.reusable(prompt);
            if (skip > 0) {
                // 保存しておいた past_* を書き戻す
                for (const auto& c : cache_in) {
                    const std::string nm = "past_" + c.second;
                    auto it = std::find(carry.cache_names.begin(), carry.cache_names.end(), nm);
                    if (it == carry.cache_names.end()) { skip = 0; break; }
                    const size_t k = static_cast<size_t>(it - carry.cache_names.begin());
                    if (carry.caches[k].size() != inputs[c.first].size()) { skip = 0; break; }
                    inputs[c.first] = carry.caches[k];
                }
            }
        }
        if (skip == 0) carry.reset();

        // ---- 位置0 の K/V を事前計算したものから始める（pos0_kv、EdgeSLM_AndroidScale S-3）----
        //
        // 1.5B は層0 のスコアが行ごとに −10,800 までずれるヘッドがあり、空きスロットを 0 で埋めると
        // −60 のマスクが効かず FP32 の時点で壊れる。位置0（<|im_start|>）の K/V を PC で正しく計算し、
        // **全スロットへ複製**して始める。端末は位置0 を計算しない。
        // ファイルは past_* の入力順に [KVH][HD] float32 を並べたもの。
        size_t pos0_skip = 0;
        // ★`pin_pos0`（Phase 3）: 窓が満杯になった後も位置0 を残す。そのために**毎ターン**読む（57KB）
        // ★`pos0_kv` のファイルを持たないモデル（0.5B など、位置0 を端末で計算する）でも使える（2026-09-19）。
        //   1ターン目の最初のステップ（位置0 の <|im_start|>）で present に入った K/V を取っておき、持ち越す。
        //   位置0 は自分自身しか見ないので、この K/V は会話の中身によらない
        const bool has_p0_file = job.contains("pos0_kv");
        const bool pin_pos0 = job.value("pin_pos0", false);
        std::vector<uint8_t> p0;
        std::vector<std::pair<int, size_t>> p0_at;     // (入力番号, p0 内の先頭)。pin_pos0 用
        size_t p0_need = 0;
        for (const auto& c : cache_in) {
            const auto& shp = d["inputs"][c.first]["shape"];          // [1, KVH, L, HD]
            p0_at.emplace_back(c.first, p0_need);
            p0_need += static_cast<size_t>(shp[1].get<int64_t>() * shp[3].get<int64_t>()) * 4;
        }
        bool p0_capture = false;                       // このターンの最初のステップで位置0 の K/V を取るか
        if ((skip == 0 || pin_pos0) && has_p0_file) {
            std::string e;
            if (!read_all(session.resolve_path(job["pos0_kv"].get<std::string>()), &p0, &e)) {
                out["ok"] = false;
                out["error"] = "generate_kv: " + e;
                return out;
            }
            if (p0_need != p0.size()) {
                out["ok"] = false;
                out["error"] = "generate_kv: pos0_kv size mismatch (need " + std::to_string(p0_need) +
                               " bytes, got " + std::to_string(p0.size()) + ")";
                return out;
            }
        } else if (pin_pos0) {
            if (skip > 0 && carry.pos0.size() == p0_need) p0 = carry.pos0;   // 1ターン目に取っておいたもの
            else if (skip == 0) p0_capture = true;
        }
        if (skip == 0 && job.contains("pos0_kv")) {
            std::string e;
            if (prompt[0] != job.value("pos0_token", static_cast<int64_t>(151644))) {
                out["ok"] = false;
                out["error"] = "generate_kv: pos0_kv needs prompt_ids[0] == <|im_start|>";
                return out;
            }
            size_t off = 0;
            for (const auto& c : cache_in) {
                const auto& shp = d["inputs"][c.first]["shape"];      // [1, KVH, L, HD]
                const int64_t kvh = shp[1].get<int64_t>(), l = shp[2].get<int64_t>(),
                              hd = shp[3].get<int64_t>();
                const size_t slot = static_cast<size_t>(hd) * 4;
                if (off + static_cast<size_t>(kvh) * slot > p0.size()) { e = "pos0_kv too small"; break; }
                auto* dst = inputs[c.first].data();
                for (int64_t h = 0; h < kvh; ++h) {
                    const uint8_t* src = p0.data() + off + static_cast<size_t>(h) * slot;
                    for (int64_t j = 0; j < l; ++j)
                        std::memcpy(dst + static_cast<size_t>(h * l + j) * slot, src, slot);
                }
                off += static_cast<size_t>(kvh) * slot;
            }
            if (!e.empty() || off != p0.size()) {
                out["ok"] = false;
                out["error"] = "generate_kv: pos0_kv size mismatch " + e;
                return out;
            }
            pos0_skip = 1;
        }

        GenProgress& prog = GenProgress::instance();
        prog.begin(max_new, static_cast<int>(L), prompt.size());

        std::vector<int64_t> newly;
        std::vector<double> step_ms, ort_ms;
        std::vector<std::vector<uint8_t>> outs;
        std::string gerr;
        bool stopped_by_eos = false;

        // ★`zero_copy`（EdgeSLM_AndroidScale Phase 6）: CPU 側の無駄を減らす。**job に書いたときだけ**効く。
        //   従来は毎ステップ ①ORT が出力 57本（約15MB）を新しく確保 ②それを vector へ写す
        //   ③present を past へ写す、をしていた。ここでは出力を**こちらのバッファへ直接**書かせ、
        //   KV は2組（inputs と kv_b）を用意して **past と present を毎ステップ入れ替える**（写さない）。
        const bool zero_copy = job.value("zero_copy", false);
        // ★バッファは**セッションが持つ**（ターンごとに作り直さない。大きさが合っていれば使い回す）
        std::vector<std::vector<uint8_t>>& kv_b = session.zc_kv_b();     // もう1組の KV
        std::vector<std::vector<uint8_t>>& zc_out = session.zc_out();    // KV 以外の出力（logits）の置き場
        std::vector<int> out_cache_in;                 // 出力番号 → その present が次に past として入る入力番号（KV でなければ -1）
        bool past_in_a = true;                         // いまの past は inputs（A）か kv_b（B）か
        if (zero_copy) {
            const auto& souts = session.outputs();
            if (kv_b.size() != inputs.size()) kv_b.resize(inputs.size());
            if (zc_out.size() != souts.size()) zc_out.resize(souts.size());
            out_cache_in.assign(souts.size(), -1);
            for (const auto& c : cache_in) {
                if (kv_b[c.first].size() != inputs[c.first].size()) kv_b[c.first].assign(inputs[c.first].size(), 0);
                out_cache_in[present_idx[c.second]] = c.first;
            }
            for (size_t oi = 0; oi < souts.size(); ++oi) {
                const size_t n = souts[oi].element_count() * souts[oi].element_size();
                if (out_cache_in[oi] < 0 && zc_out[oi].size() != n) zc_out[oi].assign(n, 0);
            }
        }
        // ★HTP 共有メモリ（Phase 7、job の `htp_shared_io: true`）。KV の2組・logits・小さい入力を
        //   NPU と共有するメモリに置き、推論ではコピーせずに渡す。取れなければ従来の zero_copy（ヒープ）で動く
        const bool shared_io = zero_copy && session.shared_io_enabled();
        std::vector<uint8_t*> kvA(inputs.size(), nullptr), kvB(inputs.size(), nullptr);
        std::vector<uint8_t*> in_sh(inputs.size(), nullptr);          // KV 以外の入力の共有メモリ
        std::vector<uint8_t*> obuf(out_cache_in.size(), nullptr);     // KV 以外の出力の置き場
        if (zero_copy) {
            bool ok_sh = shared_io;
            for (const auto& c : cache_in) {
                if (shared_io) {
                    kvA[c.first] = session.shared_buf("kvA_" + std::to_string(c.first), inputs[c.first].size());
                    kvB[c.first] = session.shared_buf("kvB_" + std::to_string(c.first), inputs[c.first].size());
                    ok_sh = ok_sh && kvA[c.first] && kvB[c.first];
                } else {
                    kvA[c.first] = inputs[c.first].data();
                    kvB[c.first] = kv_b[c.first].data();
                }
            }
            const auto& souts = session.outputs();
            for (size_t oi = 0; oi < obuf.size(); ++oi) {
                if (out_cache_in[oi] >= 0) continue;
                if (shared_io) {
                    obuf[oi] = session.shared_buf("out_" + std::to_string(oi),
                                                  souts[oi].element_count() * souts[oi].element_size());
                    ok_sh = ok_sh && obuf[oi];
                } else {
                    obuf[oi] = zc_out[oi].data();
                }
            }
            if (shared_io) {
                for (size_t i = 0; i < inputs.size(); ++i) {
                    if (kvA[i]) continue;
                    in_sh[i] = session.shared_buf("in_" + std::to_string(i), inputs[i].size());
                    ok_sh = ok_sh && in_sh[i];
                }
                if (!ok_sh) {
                    out["ok"] = false;
                    out["error"] = "generate_kv: failed to allocate HTP shared memory";
                    return out;
                }
                // このターンの最初の past（位置0 の複製か、前のターンの持ち越し）を共有メモリへ（ターンに1回）
                for (const auto& c : cache_in) std::memcpy(kvA[c.first], inputs[c.first].data(), inputs[c.first].size());
            }
        } else {
            // ★zero_copy でないときも「いまの past」は inputs。pin_pos0 と位置0 の取り出しが past_buf を使う
            //   （P7 で kvA/kvB を zero_copy の中だけで設定するようにしたため、ここが空のままだった。2026-09-19）
            for (const auto& c : cache_in) kvA[c.first] = inputs[c.first].data();
        }
        auto past_buf = [&](int in_idx) -> uint8_t* {
            return past_in_a ? kvA[in_idx] : kvB[in_idx];
        };
        std::vector<double> bd_prep, bd_run, bd_post;  // 1ステップの内訳（ms）: 準備 / 推論の呼び出し全体 / 後処理
        // ★**最初の1トークンだけ** top-K を残す。
        //   free-running の生成では最初の1手が軌道全体を決めるので、
        //   「Host の1位が HTP では何位か」「1位と2位の差はどれだけか」を
        //   見ないと、出力文が違うことを「能力が落ちた」と読み違える。
        //   `probe_ids` を job に書くと、その ID の logit と順位も返す
        //   （Host 側の上位 ID を入れて順位を調べるため）。
        json first_topk = nlohmann::json::array();
        json first_probe = nlohmann::json::array();
        std::vector<int64_t> probe_ids;
        if (job.contains("probe_ids") && job["probe_ids"].is_array()) {
            for (const auto& v : job["probe_ids"]) probe_ids.push_back(v.get<int64_t>());
        }
        int64_t last = prompt[0];
        int64_t n_valid = (skip > 0) ? carry.n_valid : static_cast<int64_t>(pos0_skip);
        int64_t abs_pos = (skip > 0) ? carry.pos : static_cast<int64_t>(pos0_skip);   // RoPE の絶対位置
        double prefill_ms = 0.0;

        // プロンプトを1トークンずつ食わせてキャッシュを埋め、そのまま生成へ続ける。
        //
        // ★再利用したぶんは **1つも食わせ直さない**（`skip` から始める）。
        //   最初 `skip - 1` から始めていたが、**再利用した最後のトークンは
        //   すでにキャッシュに入っている**ので、もう一度食わせると
        //   **同じトークンが二重に積まれて文脈が崩れる**。
        //   実機では 2往復目以降が「こんにちは、何かお手伝いできることが…」という
        //   会話が始まったばかりのような定型文になり、fp32（別々の答えを返す）と
        //   食い違って気付いた（2026-09-14）。
        const size_t start_step = skip + pos0_skip;     // 位置0 を事前計算したなら 1 から
        const size_t total_steps = prompt.size() + static_cast<size_t>(max_new);
        bool rope_exhausted = false;
        int64_t rope_exhausted_at = -1;
        int64_t pinned_steps = 0;                       // pin_pos0 で位置0 を書き戻した回数
        size_t fed_until = start_step;                  // 推論まで終わった（キャッシュに入った）トークンの数（先頭から）
        const double t_start = now_ms();
        for (size_t t = start_step; t < total_steps; ++t) {
            if (cancel_flag && *cancel_flag) { gerr = "cancelled"; break; }
            const double s0 = now_ms();

            const int64_t tid = (t < prompt.size()) ? prompt[t] : last;
            if (emb_idx >= 0) {
                if (tid < 0 || tid >= emb_rows) { gerr = "token id out of embed_table"; break; }
                const auto* src = reinterpret_cast<const uint16_t*>(emb_tab.p) + tid * H;
                auto* dst = reinterpret_cast<float*>(inputs[emb_idx].data());
                if (emb_bf16) {
                    // bf16 は fp32 の上位16ビットそのもの。重み（bf16）とビット単位で同じ値になる
                    for (int64_t k = 0; k < H; ++k) {
                        const uint32_t f = static_cast<uint32_t>(src[k]) << 16;
                        std::memcpy(dst + k, &f, 4);
                    }
                } else {
                    for (int64_t k = 0; k < H; ++k) dst[k] = half_to_float(src[k]);
                }
            } else {
                *reinterpret_cast<int64_t*>(inputs[ids_idx].data()) = tid;
            }

            // ★位置は **絶対位置のまま進める**（2026-09-14 に L-1 の張り付きをやめた）。
            //
            //   RoPE は Q と K の回転角の差が相対位置を表す。K はキャッシュへ入る時点で
            //   その時の絶対位置で回転済みなので、**Q の位置を増やし続ける限り
            //   相対位置は正しいまま**である。スロット番号が 0 側へずれても関係ない。
            //   逆に L-1 で張り付かせると、満杯以降のトークンが**全部同じ位置**になり、
            //   「何番目か」が区別できなくなる。
            //
            //   テーブルの行数で頭打ちにする（容量 L ではない）。
            // ★位置は `t` ではなく**引き継いだ絶対位置**を使う。
            //   再利用したときは `t` がプロンプト内の添字に戻るので、`t` のままだと
            //   前ターンで進めた位置を巻き戻してしまう。
            const int64_t abs_this = abs_pos;              // このトークンの絶対位置（pin_pos0 の判定用）
            const int64_t pos = std::min<int64_t>(abs_pos, rope_rows - 1);
            ++abs_pos;
            if (pos == rope_rows - 1 && !rope_exhausted) {
                rope_exhausted = true;   // 記録用。ここから先は位置が近似になる
                rope_exhausted_at = static_cast<int64_t>(t);
            }
            std::memcpy(inputs[cos_idx].data(), cos_tab.data() + row_bytes * pos, row_bytes);
            std::memcpy(inputs[sin_idx].data(), sin_tab.data() + row_bytes * pos, row_bytes);

            // ★位置0 を窓に残す（`pin_pos0`、EdgeSLM_AndroidScale Phase 3）。
            //   キャッシュは古い順に押し出すシフト式で、満杯になると最初に消えるのが位置0 の
            //   `<|im_start|>`（1.5B の注意の吸い込み口）。このステップで slot 0 が押し出され、
            //   slot 1 が slot 0 に来るので、slot 1 に位置0 の K/V を書いておく。
            //   窓は「位置0 ＋ 直近 L−1」になる（押し出されるのは位置0 の次に古いもの）。
            //   位置0 の K は位置0 で回転済みのまま置くので、Q との相対位置は正しい。
            if (pin_pos0 && abs_this >= L && p0.size() == p0_need) {
                for (const auto& pa : p0_at) {
                    const auto& shp = d["inputs"][pa.first]["shape"];
                    const int64_t kvh = shp[1].get<int64_t>(), l = shp[2].get<int64_t>(),
                                  hd = shp[3].get<int64_t>();
                    const size_t slot = static_cast<size_t>(hd) * 4;
                    auto* dst = past_buf(pa.first);            // いまの past（zero_copy では入れ替わる）
                    for (int64_t h = 0; h < kvh; ++h)
                        std::memcpy(dst + static_cast<size_t>(h * l + 1) * slot,
                                    p0.data() + pa.second + static_cast<size_t>(h) * slot, slot);
                }
                ++pinned_steps;
            }

            // マスク: 末尾 n_valid 個だけ開ける
            n_valid = std::min<int64_t>(n_valid + 1, L);
            {
                auto* mk = reinterpret_cast<float*>(inputs[mask_idx].data());
                for (int64_t j = 0; j < L; ++j) mk[j] = mask_neg;
                for (int64_t j = L - n_valid; j < L; ++j) mk[j] = 0.0f;
            }

            double ms = 0.0;
            const double tp = now_ms();
            const float* row = nullptr;
            if (zero_copy) {
                // past はいまの組、present はもう一方の組へ直接書かせる。書き終わったら役を入れ替える
                std::vector<const void*> ip(inputs.size());
                for (size_t i = 0; i < inputs.size(); ++i) {
                    if (in_sh[i]) {                      // 小さい入力（埋め込み・cos/sin・マスク。数 KB）は共有メモリへ写す
                        std::memcpy(in_sh[i], inputs[i].data(), inputs[i].size());
                        ip[i] = in_sh[i];
                    } else {
                        ip[i] = inputs[i].data();
                    }
                }
                for (const auto& c : cache_in) ip[c.first] = past_buf(c.first);
                std::vector<void*> op(out_cache_in.size());
                for (size_t oi = 0; oi < op.size(); ++oi) {
                    const int ci = out_cache_in[oi];
                    op[oi] = (ci >= 0) ? (past_in_a ? kvB[ci] : kvA[ci]) : obuf[oi];
                }
                if (!session.run_into(ip, op, &ms, shared_io)) { gerr = session.last_error(); break; }
                past_in_a = !past_in_a;                  // いま書いた present が次の past
                row = reinterpret_cast<const float*>(obuf[logits_idx]);
            } else {
                if (!session.run(inputs, &ms, &outs)) { gerr = session.last_error(); break; }
                if (outs.size() <= static_cast<size_t>(logits_idx)) { gerr = "no logits"; break; }

                // present_* を past_* へ戻す。**これが decode ループの本体**である。
                for (const auto& c : cache_in) {
                    const int oi = present_idx[c.second];
                    if (static_cast<size_t>(oi) >= outs.size() ||
                        outs[oi].size() != inputs[c.first].size()) {
                        gerr = "cache size mismatch for " + c.second;
                        break;
                    }
                    std::memcpy(inputs[c.first].data(), outs[oi].data(), outs[oi].size());
                }
                if (!gerr.empty()) break;
                row = reinterpret_cast<const float*>(outs[logits_idx].data());
            }
            const double tr = now_ms();
            bd_prep.push_back(tp - s0);
            bd_run.push_back(tr - tp);
            fed_until = t + 1;                             // ここまでのトークンはキャッシュに入った
            // ★位置0 の K/V を取っておく（pin_pos0 で pos0_kv のファイルが無いとき。1ターン目の最初のステップだけ）。
            //   新しいトークンの K/V は present の最後のスロット（L−1）に入る。past_buf はいま書いた present を指す
            if (p0_capture && t == 0) {
                p0.assign(p0_need, 0);
                for (const auto& pa : p0_at) {
                    const auto& shp = d["inputs"][pa.first]["shape"];
                    const int64_t kvh = shp[1].get<int64_t>(), l = shp[2].get<int64_t>(),
                                  hd = shp[3].get<int64_t>();
                    const size_t slot = static_cast<size_t>(hd) * 4;
                    const uint8_t* src = past_buf(pa.first);
                    for (int64_t h = 0; h < kvh; ++h)
                        std::memcpy(p0.data() + pa.second + static_cast<size_t>(h) * slot,
                                    src + static_cast<size_t>(h * l + (l - 1)) * slot, slot);
                }
                p0_capture = false;
            }
            int64_t best = 0;
            float bestv = row[0];
            for (int64_t k = 1; k < V; ++k) {   // 同点は先勝ち（numpy.argmax と同じ）
                if (row[k] > bestv) { bestv = row[k]; best = k; }
            }
            // ★最初の1トークンを出す回だけ記録する
            if (t == prompt.size() - 1) {
                const int K = 10;
                std::vector<int64_t> top;
                std::vector<bool> used(0);
                for (int r = 0; r < K; ++r) {
                    int64_t bi = -1; float bv = 0.0f;
                    for (int64_t k = 0; k < V; ++k) {
                        bool skip_k = false;
                        for (int64_t u : top) if (u == k) { skip_k = true; break; }
                        if (skip_k) continue;
                        if (bi < 0 || row[k] > bv) { bv = row[k]; bi = k; }
                    }
                    if (bi < 0) break;
                    top.push_back(bi);
                    first_topk.push_back({{"id", bi}, {"logit", bv}});
                }
                for (int64_t pid : probe_ids) {
                    if (pid < 0 || pid >= V) continue;
                    // 順位 = 自分より大きい値の個数（同点は先勝ちに合わせて添字で割る）
                    int64_t rank = 0;
                    for (int64_t k = 0; k < V; ++k) {
                        if (row[k] > row[pid] || (row[k] == row[pid] && k < pid)) ++rank;
                    }
                    first_probe.push_back({{"id", pid}, {"logit", row[pid]}, {"rank", rank}});
                }
            }

            const double s1 = now_ms();
            bd_post.push_back(s1 - tr);

            if (t < prompt.size() - 1) {
                // まだプロンプトを食べている途中。出力は使わない
                prefill_ms += (s1 - s0);
                continue;
            }
            last = best;
            newly.push_back(best);
            step_ms.push_back(s1 - s0);
            ort_ms.push_back(ms);
            prog.push(best, s1 - s0, ms);
            if (std::find(stop_ids.begin(), stop_ids.end(), best) != stop_ids.end()) {
                stopped_by_eos = true;
                break;
            }
            if (static_cast<int>(newly.size()) >= max_new) break;
        }
        const double total_ms = now_ms() - t_start;
        if (zero_copy && (shared_io || !past_in_a)) {
            // 最後の past を inputs 側へ写す（carry は inputs を読む。ターンに1回）。**入れ替え（swap）はしない**——
            // swap するとセッションのバッファがジョブの vector と入れ替わり、次のターンでアドレスが変わる
            for (const auto& c : cache_in)
                std::memcpy(inputs[c.first].data(), past_buf(c.first), inputs[c.first].size());
            past_in_a = true;
        }
        prog.finish(gerr);

        // ---- 次のターンのためにキャッシュを保存する ----
        //
        // ★**失敗したときは保存しない。** 途中で落ちたキャッシュを引き継ぐと、
        //   次のターンが静かに壊れる（何が起きたか分からなくなる）。
        // ★ただし**止めた（cancel）ときは保存する**（2026-09-19）。キャッシュは壊れておらず、`fed_until` までが入っている。
        //   保存しないと、画面は次のターンで差分だけを送るので、1.5B は「pos0_kv needs prompt_ids[0] == <|im_start|>」で
        //   失敗し、0.5B は履歴なしで答えていた（チャット画面・KV 版の画面とも）
        const bool cancelled = (gerr == "cancelled");
        if ((gerr.empty() || cancelled) && want_reuse) {
            carry.model = d.value("model", std::string());
            carry.cache_names.clear();
            carry.caches.clear();
            for (const auto& c : cache_in) {
                carry.cache_names.push_back("past_" + c.second);
                carry.caches.push_back(inputs[c.first]);
            }
            carry.n_valid = n_valid;
            carry.pos = abs_pos;
            // ★**実際にキャッシュへ入ったトークンだけ**を記録する。
            //
            //   生成ループは「トークンを選ぶ → 次の周回で食わせる」形なので、
            //   **最後に選んだ1つは食わせずに終わる**（停止判定や max_new で抜けるため）。
            //   つまりキャッシュにあるのは `prompt + newly[0 .. n-2]` である。
            //
            //   ここで `newly` を丸ごと入れると、次のターンで「入っている」と誤認して
            //   そのトークンを飛ばし、**文脈から1つ抜け落ちる**。
            //   実機では 2往復目以降の応答が「毎回フル入力」と食い違って気付いた
            //   （2026-09-14。差分＋再利用と毎回フル入力は**同じ応答になるはず**）。
            //   止めたときも同じ考え方で、推論まで終わった `fed_until` 個までを記録する
            //   （最後まで生成したときは fed_until = prompt + newly − 1 で、上の規則と同じになる）。
            carry.tokens = prompt;
            carry.tokens.insert(carry.tokens.end(), newly.begin(), newly.end());
            if (carry.tokens.size() > fed_until) carry.tokens.resize(fed_until);
            if (pin_pos0 && !has_p0_file && p0.size() == p0_need) carry.pos0 = p0;
            carry.valid = true;
        } else if (!gerr.empty()) {
            carry.reset();
        }

        out["new_ids"] = newly;
        out["step_ms"] = step_ms;
        out["ort_run_ms"] = stats_of(ort_ms);
        out["step_ms_stats"] = stats_of(step_ms);
        out["zero_copy"] = zero_copy;
        out["htp_shared_io"] = shared_io;
        out["breakdown_ms"] = {{"prep", stats_of(bd_prep)}, {"run_wall", stats_of(bd_run)},
                               {"post", stats_of(bd_post)}};   // プロンプトの充填も含む全ステップ
        {
            json all = json::array();                  // 全ステップの推論の呼び出し時間（充填も含む。診断用）
            for (double v : bd_run) all.push_back(std::round(v * 100.0) / 100.0);
            out["run_wall_all_ms"] = all;
        }
        out["total_ms"] = total_ms;
        out["prefill_ms"] = prefill_ms;      // キャッシュを埋めるのに使った時間（内訳）
        // ★最初の1トークンの top-10 と、指定 ID の logit / 順位（Host と突き合わせる用）
        if (!first_topk.empty()) out["first_topk"] = first_topk;
        if (!first_probe.empty()) out["first_probe"] = first_probe;
        out["reused_tokens"] = static_cast<int64_t>(skip);   // 充填を飛ばせたトークン数
        out["pos0_preloaded"] = pos0_skip > 0;
        out["pin_pos0"] = pin_pos0;
        out["pinned_steps"] = pinned_steps;
        out["pin_pos0_src"] = !pin_pos0 ? "off" : has_p0_file ? "file"
                            : (p0.size() == p0_need ? "captured" : "none");   // 位置0 の K/V をどこから得たか
        out["carry_tokens"] = carry.valid ? static_cast<int64_t>(carry.tokens.size()) : 0;   // 次のターンへ持ち越した数
        out["abs_pos"] = abs_pos;                       // 次に食わせるトークンの絶対位置（L を超えたら窓が滑っている）
        out["prefill_tokens"] = static_cast<int64_t>(
            prompt.size() > skip ? prompt.size() - skip : 0);
        out["cache_len"] = L;
        out["rope_rows"] = rope_rows;
        out["rope_exhausted_at"] = rope_exhausted_at;   // -1 なら最後まで正しい位置だった
        out["prompt_len"] = prompt.size();
        out["cache_tensors"] = cache_in.size();
        out["stopped_by_eos"] = stopped_by_eos;
        if (!gerr.empty()) {
            out["ok"] = false;
            out["error"] = gerr;
        } else {
            out["ok"] = !newly.empty();
        }
        return out;
    }

    if (task == "op_check") {
        // P2b: 単体 op モデルを次々に開いて「どこまで通るか」を判定する。
        //
        // **APK を作り直さずに対象を足せる**ことが肝で、models は job.json 側にある。
        // 判定は「通る / 通らない」の2値にせず、次の手が決まる粒度まで割る:
        //   pass_strict        … 全ノードが HTP。そのまま使える
        //   pass_cpu_boundary  … 開けたが CPU フォールバックあり。delegate しきれていない
        //   open_failed        … ロードで落ちた（C-1〜C-7 / op 非対応）
        //   run_failed         … 開けたが実行で落ちた
        json results = json::array();
        const json models = job.value("models", json::array());
        const json proto = job.value("protocol", json::object());
        const int warmup = proto.value("warmup", 2);
        const int iters  = proto.value("iterations", 5);
        int n_strict = 0, n_boundary = 0, n_failed = 0;

        for (const auto& entry : models) {
            const bool is_str = entry.is_string();
            const std::string path = is_str ? entry.get<std::string>()
                                            : entry.value("model", std::string());
            const std::string label = is_str ? path : entry.value("name", path);

            json r;
            r["name"] = label;
            r["model"] = path;

            if (cancel_flag && *cancel_flag) {
                r["verdict"] = "cancelled";
                results.push_back(r);
                break;
            }

            if (!session.reopen(path)) {
                r["verdict"] = "open_failed";
                r["error"] = session.last_error();
                results.push_back(r);
                ++n_failed;
                LOGI("op_check %s: open_failed", label.c_str());
                continue;
            }
            const json desc = session.describe();
            r["describe"] = desc;
            const bool strict = desc.value("strict_no_cpu_fallback", false);
            r["strict_no_cpu_fallback"] = strict;

            const json in_spec = is_str ? job.value("input", json::object())
                                        : entry.value("input", job.value("input", json::object()));
            std::vector<std::vector<uint8_t>> inputs;
            std::string err;
            if (!build_inputs(session, in_spec, &inputs, &err)) {
                r["verdict"] = "run_failed";
                r["error"] = "input: " + err;
                results.push_back(r);
                ++n_failed;
                continue;
            }

            bool run_ok = true;
            std::vector<double> warm, samples;
            for (int i = 0; i < warmup && run_ok; ++i) {
                double ms = 0.0;
                run_ok = session.run(inputs, &ms);
                if (run_ok) warm.push_back(ms);
            }
            for (int i = 0; i < iters && run_ok; ++i) {
                if (cancel_flag && *cancel_flag) break;
                double ms = 0.0;
                run_ok = session.run(inputs, &ms);
                if (run_ok) samples.push_back(ms);
            }
            r["warmup_ms"] = warm;
            r["samples_ms"] = samples;
            if (!samples.empty()) r["ort_run_ms"] = stats_of(samples);

            if (!run_ok) {
                r["verdict"] = "run_failed";
                r["error"] = session.last_error();
                ++n_failed;
            } else if (strict) {
                r["verdict"] = "pass_strict";
                ++n_strict;
            } else {
                r["verdict"] = "pass_cpu_boundary";
                ++n_boundary;
            }
            LOGI("op_check %s: %s", label.c_str(),
                 r["verdict"].get<std::string>().c_str());
            results.push_back(r);
        }

        out["results"] = results;
        out["summary"] = {{"pass_strict", n_strict},
                          {"pass_cpu_boundary", n_boundary},
                          {"failed", n_failed},
                          {"total", results.size()}};
        // describe は「最後に開いたモデル」のものになってしまい誤解を招くので外す。
        out.erase("describe");
        out["ok"] = (n_failed == 0) && !results.empty();
        return out;
    }

    // ---- bench ----
    const json proto = job.value("protocol", json::object());
    const int warmup = proto.value("warmup", 3);
    const int iters  = proto.value("iterations", 10);

    std::vector<uint8_t> input;
    std::string err;
    if (!build_input(session, job.value("input", json::object()), &input, &err)) {
        out["ok"] = false;
        out["error"] = err;
        return out;
    }
    out["input_bytes"] = input.size();

    // ウォームアップ分は破棄する。ただし「破棄した」ことと値は記録する。
    // 初回が遅いのは既知（EdgeSLM の知見）で、その事実自体が情報だからである。
    std::vector<double> warm_ms;
    for (int i = 0; i < warmup; ++i) {
        if (cancel_flag && *cancel_flag) { out["cancelled"] = true; break; }
        double ms = 0.0;
        if (!session.run(input, &ms)) {
            out["ok"] = false;
            out["error"] = "warmup run failed: " + session.last_error();
            out["warmup_ms"] = warm_ms;
            return out;
        }
        warm_ms.push_back(ms);
        LOGI("warmup[%d] OrtRun %.2f ms", i, ms);
    }
    out["warmup_ms"] = warm_ms;

    std::vector<double> samples;
    for (int i = 0; i < iters; ++i) {
        if (cancel_flag && *cancel_flag) { out["cancelled"] = true; break; }
        double ms = 0.0;
        if (!session.run(input, &ms)) {
            out["ok"] = false;
            out["error"] = "run failed at iteration " + std::to_string(i) + ": " +
                           session.last_error();
            out["samples_ms"] = samples;
            return out;
        }
        samples.push_back(ms);
        LOGI("bench[%d] OrtRun %.2f ms", i, ms);
    }

    // 生サンプルを必ず残す（基本設計① 5章）。統計量だけでは
    // 「外れ値1個に引きずられていた」ことを後から検証できない。
    out["samples_ms"] = samples;
    out["ort_run_ms"] = stats_of(samples);
    out["ok"] = !samples.empty();
    return out;
}

}  // namespace edgeslm
