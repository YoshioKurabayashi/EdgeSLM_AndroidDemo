#include "runner/memprobe.h"

#include <android/log.h>
#include <dirent.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <new>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "runner/session.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "EdgeSLMRunner", __VA_ARGS__)

namespace edgeslm {

using nlohmann::json;

namespace {

double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

std::string slurp(const char* path) {
    std::ifstream f(path);
    if (!f) return {};
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// "Key:   12345 kB" の形の行から数値を取る。**行頭の完全一致**で探す
// （"Pss:" で探して "Pss_Anon:" を拾わないように）。無ければ -1。
long long field(const std::string& text, const char* key) {
    const std::string k = std::string(key) + ":";
    size_t pos = 0;
    while ((pos = text.find(k, pos)) != std::string::npos) {
        if (pos == 0 || text[pos - 1] == '\n') return std::atoll(text.c_str() + pos + k.size());
        pos += k.size();
    }
    return -1;
}

// このプロセスが fd で保持している dma-buf の合計。
// FastRPC / ION の共有バッファはここに現れ、**RSS には必ずしも入らない**。
// /proc/self/fdinfo/<fd> に exp_name: と size: が出るものを dma-buf とみなす。
json dmabuf_usage() {
    long long bytes = 0;
    int count = 0;
    DIR* d = opendir("/proc/self/fdinfo");
    if (!d) return json{{"bytes", -1}, {"count", -1}};
    while (dirent* e = readdir(d)) {
        if (e->d_name[0] == '.') continue;
        const std::string p = std::string("/proc/self/fdinfo/") + e->d_name;
        const std::string t = slurp(p.c_str());
        if (t.find("exp_name:") == std::string::npos) continue;
        const long long sz = field(t, "size");
        if (sz > 0) {
            bytes += sz;
            ++count;
        }
    }
    closedir(d);
    return json{{"bytes", bytes}, {"count", count}};
}

json snapshot(const char* label) {
    const std::string st = slurp("/proc/self/status");
    const std::string mi = slurp("/proc/meminfo");
    const std::string sr = slurp("/proc/self/smaps_rollup");
    json j;
    j["label"] = label;
    j["t_ms"] = now_ms();
    j["vm_rss_kb"] = field(st, "VmRSS");
    j["vm_hwm_kb"] = field(st, "VmHWM");   // 直前の reset_hwm() からの最大 RSS
    j["vm_size_kb"] = field(st, "VmSize");
    j["vm_swap_kb"] = field(st, "VmSwap");
    j["rss_anon_kb"] = field(st, "RssAnon");
    j["rss_file_kb"] = field(st, "RssFile");
    j["rss_shmem_kb"] = field(st, "RssShmem");
    j["pss_kb"] = sr.empty() ? -1 : field(sr, "Pss");
    j["mem_available_kb"] = field(mi, "MemAvailable");
    j["mem_free_kb"] = field(mi, "MemFree");
    j["swap_free_kb"] = field(mi, "SwapFree");
    j["cma_free_kb"] = field(mi, "CmaFree");
    j["dmabuf"] = dmabuf_usage();
    return j;
}

// VmHWM（最大 RSS）を今の RSS に戻す。**1本のロード中の山**を測るため。
// clear_refs への書き込みが許されない環境では false（その場合 HWM は単調増加のまま）。
bool reset_hwm() {
    FILE* f = std::fopen("/proc/self/clear_refs", "w");
    if (!f) return false;
    const bool ok = std::fputs("5", f) >= 0;
    return (std::fclose(f) == 0) && ok;
}

// 一定間隔で RSS 等を CSV に追記する。**殺される直前まで**の推移を残すのが目的。
class Sampler {
public:
    Sampler(const std::string& path, int interval_ms) : interval_ms_(interval_ms) {
        fp_ = std::fopen(path.c_str(), "w");
        if (!fp_) return;
        std::fputs("t_ms,step,vm_rss_kb,vm_hwm_kb,rss_anon_kb,rss_file_kb,mem_available_kb,"
                   "swap_free_kb,dmabuf_bytes\n", fp_);
        std::fflush(fp_);
        th_ = std::thread([this] { loop(); });
    }
    ~Sampler() {
        stop_.store(true);
        if (th_.joinable()) th_.join();
        if (fp_) std::fclose(fp_);
    }
    void set_step(int s) { step_.store(s); }

private:
    void loop() {
        while (!stop_.load()) {
            const std::string st = slurp("/proc/self/status");
            const std::string mi = slurp("/proc/meminfo");
            const json db = dmabuf_usage();
            std::fprintf(fp_, "%.0f,%d,%lld,%lld,%lld,%lld,%lld,%lld,%lld\n", now_ms(),
                         step_.load(), field(st, "VmRSS"), field(st, "VmHWM"),
                         field(st, "RssAnon"), field(st, "RssFile"), field(mi, "MemAvailable"),
                         field(mi, "SwapFree"), db.value("bytes", -1LL));
            std::fflush(fp_);
            std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_));
        }
    }

    FILE* fp_ = nullptr;
    int interval_ms_;
    std::atomic<bool> stop_{false};
    std::atomic<int> step_{0};
    std::thread th_;
};

// 入力はすべてゼロ。値は問わない（**確保と実行が起きること**だけを見る）。
bool zero_inputs(const Session& s, std::vector<std::vector<uint8_t>>* out, std::string* err) {
    out->clear();
    for (const auto& in : s.inputs()) {
        const size_t n = in.element_count();
        const size_t e = in.element_size();
        if (n == 0 || e == 0) {
            *err = "input '" + in.name + "' has dynamic shape or unknown type";
            return false;
        }
        out->emplace_back(n * e, 0);
    }
    return true;
}

}  // namespace

// ロード前に**ゆっくり**メモリを確保して触り、解放する。
// 目的: カーネルにファイルキャッシュを少しずつ回収させ、MemFree を増やしておく。
//
// なぜ要るか（2026-09-19 実測）: モデルのロードは 1〜2 秒で
//   読み込みバッファ（binary と同サイズの RSS）＋ページキャッシュ＋dma-buf
// を一気に要求する。MemFree が少ない（ファイルキャッシュに取られている）状態だと
// 回収が間に合わず low watermark を割り、lmkd がバックグラウンドのアプリから順に kill し、
// 最後にロード中の自分自身（adj 200）を "device is not responding" で kill する。
// MemAvailable が 3.5GB あっても起きる（効くのは MemFree）。
json prereclaim(long long mb, long long step_mb, int sleep_ms) {
    json r;
    r["before"] = snapshot("prereclaim_before");
    std::vector<std::unique_ptr<char[]>> chunks;
    const size_t step = static_cast<size_t>(step_mb) * 1024 * 1024;
    long long done = 0;
    const double t0 = now_ms();
    while (done + step_mb <= mb) {
        std::unique_ptr<char[]> c(new (std::nothrow) char[step]);
        if (!c) break;
        // 1ページずつ触って実体を割り当てさせる（触らないと確保されない）
        for (size_t i = 0; i < step; i += 4096) c[i] = 1;
        chunks.push_back(std::move(c));
        done += step_mb;
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
    }
    r["allocated_mb"] = done;
    r["alloc_ms"] = now_ms() - t0;
    r["peak"] = snapshot("prereclaim_peak");
    chunks.clear();
    r["after"] = snapshot("prereclaim_after");
    return r;
}

json run_memprobe(const json& job) {
    json out;
    const std::string job_id = job.value("job_id", std::string("memprobe"));
    out["job_id"] = job_id;
    out["task"] = "memprobe";
    out["power_profile"] = "default(unsigned_pd)";
    out["versions"] = Session::versions();
    out["pid"] = static_cast<int>(getpid());

    const std::string base = job.value("runtime", json::object()).value("base_dir", std::string());
    if (base.empty()) {
        out["ok"] = false;
        out["error"] = "runtime.base_dir is empty (RunnerService injects it)";
        return out;
    }
    const std::string steps_rel = "out/memprobe_" + job_id + "_steps.jsonl";
    const std::string samples_rel = "out/memprobe_" + job_id + "_samples.csv";
    out["steps_file"] = steps_rel;
    out["samples_file"] = samples_rel;

    FILE* steps = std::fopen((base + "/" + steps_rel).c_str(), "w");
    auto emit = [&](const json& j) {
        const std::string line = j.dump();
        if (steps) {
            std::fputs((line + "\n").c_str(), steps);
            std::fflush(steps);
        }
        LOGI("memprobe %s", line.c_str());
    };

    const json models = job.value("models", json::array());
    const bool hold = job.value("hold", true);
    const bool run_once = job.value("run_once", true);
    const int settle_ms = job.value("settle_ms", 1000);
    const bool stop_on_fail = job.value("stop_on_fail", true);

    Sampler sampler(base + "/" + samples_rel, job.value("sample_ms", 100));

    const bool hwm_ok = reset_hwm();
    out["hwm_reset_supported"] = hwm_ok;
    const json base_snap = snapshot("baseline");
    out["baseline"] = base_snap;
    emit({{"step", "baseline"}, {"pid", static_cast<int>(getpid())}, {"hwm_reset_supported", hwm_ok},
          {"hold", hold}, {"n_models", models.size()}, {"snap", base_snap}});

    std::vector<std::unique_ptr<Session>> held;
    json records = json::array();
    int index = 0;
    for (const auto& m : models) {
        const std::string rel = m.get<std::string>();
        sampler.set_step(index + 1);
        json rec;
        rec["index"] = index;
        rec["model"] = rel;

        // 各ロードの前に prereclaim（指定があれば）。0 なら何もしない
        const long long pre_mb = job.value("prereclaim_mb", 0LL);
        if (pre_mb > 0) {
            const json pr = prereclaim(pre_mb, job.value("prereclaim_step_mb", 32LL),
                                       job.value("prereclaim_sleep_ms", 50));
            rec["prereclaim"] = pr;
            emit({{"step", "prereclaim"}, {"index", index}, {"allocated_mb", pr["allocated_mb"]},
                  {"alloc_ms", pr["alloc_ms"]}, {"snap", pr["after"]}, {"peak", pr["peak"]}});
        }

        reset_hwm();
        rec["before"] = snapshot("before");
        emit({{"step", "open_begin"}, {"index", index}, {"model", rel}, {"held", held.size()},
              {"snap", rec["before"]}});

        auto s = std::make_unique<Session>();
        json cfg = job;
        cfg["model"] = rel;
        cfg.erase("prereclaim_mb");        // ここで済ませたので Session::open では繰り返さない
        const double t0 = now_ms();
        const bool ok = s->open(cfg);
        rec["open_ms"] = now_ms() - t0;
        rec["open_ok"] = ok;
        rec["after_open"] = snapshot("after_open");   // vm_hwm_kb がロード中の山
        if (ok) rec["backend"] = s->backend_label();
        else rec["error"] = s->last_error();
        emit({{"step", "open_end"}, {"index", index}, {"ok", ok}, {"open_ms", rec["open_ms"]},
              {"backend", ok ? s->backend_label() : std::string()},
              {"error", ok ? std::string() : s->last_error()}, {"snap", rec["after_open"]}});

        if (ok && run_once) {
            std::vector<std::vector<uint8_t>> inputs;
            std::string err;
            double ms = 0.0;
            const bool built = zero_inputs(*s, &inputs, &err);
            const bool ran = built && s->run(inputs, &ms);
            rec["run_ok"] = ran;
            rec["run_ms"] = ms;
            if (!ran) rec["run_error"] = built ? s->last_error() : err;
            rec["after_run"] = snapshot("after_run");
            emit({{"step", "run_end"}, {"index", index}, {"ok", ran}, {"run_ms", ms},
                  {"snap", rec["after_run"]}});
        }

        if (settle_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(settle_ms));
            rec["settled"] = snapshot("settled");
            emit({{"step", "settled"}, {"index", index}, {"snap", rec["settled"]}});
        }

        if (ok && hold) {
            held.push_back(std::move(s));
        } else {
            s.reset();
            rec["after_release"] = snapshot("after_release");
            emit({{"step", "released"}, {"index", index}, {"snap", rec["after_release"]}});
        }
        records.push_back(rec);
        ++index;
        if (!ok && stop_on_fail) break;
    }

    // hold_ms: 全部開いたまま待つ。**別プロセスから同時に NPU を使う**試験のため
    // （上限が「プロセスごと」か「DSP 全体」かを分ける）。
    const int hold_ms = job.value("hold_ms", 0);
    if (hold_ms > 0) {
        emit({{"step", "holding"}, {"hold_ms", hold_ms}, {"held", held.size()},
              {"snap", snapshot("holding")}});
        std::this_thread::sleep_for(std::chrono::milliseconds(hold_ms));
        emit({{"step", "hold_end"}, {"held", held.size()}, {"snap", snapshot("hold_end")}});
    }

    // 保持していたものを新しい順に閉じる。閉じたときに何がどれだけ戻るかも記録する
    sampler.set_step(-1);
    json releases = json::array();
    while (!held.empty()) {
        held.pop_back();
        releases.push_back(snapshot("release_one"));
    }
    out["records"] = records;
    out["releases"] = releases;
    out["final"] = snapshot("final");
    out["ok"] = true;
    emit({{"step", "done"}, {"n_opened", index}, {"snap", out["final"]}});
    if (steps) std::fclose(steps);
    return out;
}

}  // namespace edgeslm
