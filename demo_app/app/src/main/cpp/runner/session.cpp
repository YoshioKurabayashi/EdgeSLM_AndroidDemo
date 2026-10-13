#include "runner/session.h"
#include "runner/memprobe.h"

#include <android/log.h>
#include <dlfcn.h>

#include <chrono>
#include <cstdlib>   // setenv / getenv（FastRPC のパス設定に使う）
#include <cstring>
#include <thread>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  "EdgeSLMRunner", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "EdgeSLMRunner", __VA_ARGS__)

namespace edgeslm {

using nlohmann::json;

namespace {

double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

const char* type_name(ONNXTensorElementDataType t) {
    switch (t) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:  return "float32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:  return "uint8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:   return "int8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return "uint16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:  return "int16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:  return "int32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:  return "int64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:return "float16";
        default: return "other";
    }
}

// job.json の相対パスを base_dir から解決する。絶対パスならそのまま。
std::string resolve(const std::string& base, const std::string& p) {
    if (!p.empty() && p[0] == '/') return p;
    if (base.empty()) return p;
    return base + "/" + p;
}

// FastRPC のライブラリ探索パスを**自分で**設定する（既定では呼ばない）。
//
// ★重要な教訓（2026-09-12 に実際に踏んだ）:
//   ORT の QNN EP は `ADSP_LIBRARY_PATH` を**自分で設定する**。こちらが先に設定すると
//   ORT は自分の設定をスキップし、次の警告を出したうえで HTP バックエンドの
//   デバイス生成に失敗する:
//     W qnn_backend_manager.cc:308] Using existing ADSP_LIBRARY_PATH setting of <path>,
//       which may cause the HTP backend to fail.
//     E qnn_execution_provider.cc:1046 GetCapability] QNN SetupBackend failed
//       Failed to create device. Error: QNN_DEVICE_ERROR_INVALID_CONFIG
//   さらにその後の症状は「EPContext node is not compatible with any execution provider」
//   という**真因を示さないエラー**になる。
//
//   → したがって既定では**何もしない**。ORT に任せる。
//      必要な場合のみ job.json の runtime.set_fastrpc_paths=true で有効化する
//      （ZipDepth は自前で設定しており、比較実験のために残してある）。
//
// 区切り文字が2つで異なる点に注意:
//   LD_LIBRARY_PATH   → ':'（POSIX 流儀）
//   ADSP_LIBRARY_PATH → ';'（Qualcomm 流儀）
void configure_fastrpc_paths(const std::string& dir) {
    if (dir.empty()) return;
    const char* old_ld = getenv("LD_LIBRARY_PATH");
    const char* old_adsp = getenv("ADSP_LIBRARY_PATH");
    if (old_ld && strstr(old_ld, dir.c_str()) &&
        old_adsp && strstr(old_adsp, dir.c_str())) {
        return;  // 既に設定済み
    }
    std::string ld = dir;
    if (old_ld && old_ld[0]) { ld += ":"; ld += old_ld; }
    setenv("LD_LIBRARY_PATH", ld.c_str(), 1);

    std::string adsp = dir;
    if (old_adsp && old_adsp[0]) { adsp += ";"; adsp += old_adsp; }
    setenv("ADSP_LIBRARY_PATH", adsp.c_str(), 1);

    LOGI("fastrpc paths configured: %s", dir.c_str());
}

// OrtStatus* を返すが失敗しても続行してよい呼び出し用。
// **黙って捨てない**——落とすと -Wunused-result が出るうえ、
// 後から「設定できていたのか」を確認できなくなる。警告として残す。
void discard(const OrtApi* api, OrtStatus* status, const char* where) {
    if (!status) return;
    const char* msg = api->GetErrorMessage(status);
    LOGI("(non-fatal) %s: %s", where, msg ? msg : "(no message)");
    api->ReleaseStatus(status);
}

}  // namespace

size_t TensorInfo::element_size() const {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:  return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:   return 1;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:  return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:  return 8;
        default: return 0;
    }
}

size_t TensorInfo::element_count() const {
    size_t n = 1;
    for (int64_t d : shape) {
        if (d <= 0) return 0;  // 動的次元。QNN EP は動的形状を扱えない（要件定義書 3.3.2節#11）
        n *= static_cast<size_t>(d);
    }
    return n;
}

Session::Session() = default;

Session::~Session() {
    if (api_) {
        free_shared();
        if (run_options_) api_->ReleaseRunOptions(run_options_);
        if (mem_info_)    api_->ReleaseMemoryInfo(mem_info_);
        if (session_)     api_->ReleaseSession(session_);
        if (env_)         api_->ReleaseEnv(env_);
    }
    // ort_lib_ は dlclose しない。QNN バックエンドはプロセス全体の状態を持つため、
    // 閉じて開き直すと不安定になる（ZipDepth も同じ判断をしている）。
}

void Session::set_error(const std::string& where, const std::string& what) {
    last_error_ = where + ": " + what;
    LOGE("%s", last_error_.c_str());
}

bool Session::check(OrtStatus* status, const std::string& where) {
    if (status == nullptr) return true;
    const char* msg = api_->GetErrorMessage(status);
    set_error(where, msg ? msg : "(no message)");
    api_->ReleaseStatus(status);
    return false;
}

bool Session::load_ort() {
    if (api_) return true;
    ort_lib_ = dlopen("libonnxruntime.so", RTLD_NOW | RTLD_LOCAL);
    if (!ort_lib_) {
        set_error("dlopen libonnxruntime.so", dlerror() ? dlerror() : "(null)");
        return false;
    }
    using GetApiBaseFn = const OrtApiBase* (*)(void);
    auto get_api_base = reinterpret_cast<GetApiBaseFn>(dlsym(ort_lib_, "OrtGetApiBase"));
    if (!get_api_base) {
        set_error("dlsym OrtGetApiBase", dlerror() ? dlerror() : "(null)");
        return false;
    }
    api_ = get_api_base()->GetApi(ORT_API_VERSION);
    if (!api_) {
        set_error("GetApi", "requested ORT C API version is unavailable");
        return false;
    }
    return true;
}

bool Session::create_session(const std::string& model_path,
                             const json& config,
                             bool strict) {
    OrtSessionOptions* options = nullptr;
    if (!check(api_->CreateSessionOptions(&options), "CreateSessionOptions")) return false;

    const json ep      = config.value("ep", json::object());
    const json sess    = ep.value("session", json::object());
    const json ep_opts = ep.value("options", json::object());

    discard(api_, api_->SetSessionLogSeverityLevel(options, sess.value("log_severity_level", 2)),
            "SetSessionLogSeverityLevel");
    discard(api_, api_->SetIntraOpNumThreads(options, sess.value("intra_op_num_threads", 1)),
            "SetIntraOpNumThreads");
    discard(api_, api_->SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL),
            "SetSessionGraphOptimizationLevel");

    // --- QNN EP のプロバイダオプション ---
    // backend_path は実ファイルパスでなければならない（C-6）。
    const std::string native_lib_dir = config.value("runtime", json::object())
                                             .value("native_lib_dir", std::string());
    const std::string backend_so = ep.value("backend", std::string("libQnnHtp.so"));
    const std::string backend_path = native_lib_dir.empty()
                                   ? backend_so
                                   : native_lib_dir + "/" + backend_so;

    std::vector<std::string> key_store, val_store;
    key_store.push_back("backend_path");
    val_store.push_back(backend_path);
    for (auto it = ep_opts.begin(); it != ep_opts.end(); ++it) {
        key_store.push_back(it.key());
        val_store.push_back(it.value().is_string() ? it.value().get<std::string>()
                                                   : it.value().dump());
    }
    // ★HTP 共有メモリ（Phase 7）。job の `htp_shared_io: true` のときだけ
    if (config.value("htp_shared_io", false)) {
        key_store.push_back("enable_htp_shared_memory_allocator");
        val_store.push_back("1");
    }
    std::vector<const char*> keys, vals;
    keys.reserve(key_store.size());
    vals.reserve(val_store.size());
    for (size_t i = 0; i < key_store.size(); ++i) {
        keys.push_back(key_store[i].c_str());
        vals.push_back(val_store[i].c_str());
    }

    OrtStatus* append = api_->SessionOptionsAppendExecutionProvider(
        options, "QNN", keys.data(), vals.data(), keys.size());
    if (!check(append, "SessionOptionsAppendExecutionProvider(QNN)")) {
        api_->ReleaseSessionOptions(options);
        return false;
    }

    // strict の場合のみ CPU フォールバックを禁止する。
    // これが成功したかどうかが、そのままグラフ配置のラベルになる。
    if (strict) {
        discard(api_, api_->AddSessionConfigEntry(options, "session.disable_cpu_ep_fallback", "1"),
                "disable_cpu_ep_fallback");
    }

    // 電力プロファイル（htp_performance_mode / rpc_control_latency）は設定しない。
    // unsigned PD では setPowerConfig が許可されず Run が失敗するため（C-3）。

    OrtStatus* st = api_->CreateSession(env_, model_path.c_str(), options, &session_);
    api_->ReleaseSessionOptions(options);
    if (st != nullptr) {
        const char* msg = api_->GetErrorMessage(st);
        // strict 失敗は「異常」ではなく判定材料なので、ログレベルを落として記録する
        LOGI("CreateSession(strict=%d) failed: %s", strict ? 1 : 0, msg ? msg : "(no message)");
        last_error_ = std::string("CreateSession(strict=") + (strict ? "1" : "0") + "): " +
                      (msg ? msg : "(no message)");
        api_->ReleaseStatus(st);
        session_ = nullptr;
        return false;
    }
    return true;
}

bool Session::collect_io() {
    OrtAllocator* alloc = nullptr;
    if (!check(api_->GetAllocatorWithDefaultOptions(&alloc), "GetAllocatorWithDefaultOptions"))
        return false;

    size_t n_in = 0, n_out = 0;
    if (!check(api_->SessionGetInputCount(session_, &n_in), "SessionGetInputCount")) return false;
    if (!check(api_->SessionGetOutputCount(session_, &n_out), "SessionGetOutputCount")) return false;

    auto collect = [&](bool is_input, size_t count, std::vector<TensorInfo>* out) -> bool {
        for (size_t i = 0; i < count; ++i) {
            char* name = nullptr;
            OrtStatus* st = is_input ? api_->SessionGetInputName(session_, i, alloc, &name)
                                     : api_->SessionGetOutputName(session_, i, alloc, &name);
            if (!check(st, "SessionGetName")) return false;

            TensorInfo info;
            info.name = name ? name : "";
            alloc->Free(alloc, name);

            OrtTypeInfo* type_info = nullptr;
            st = is_input ? api_->SessionGetInputTypeInfo(session_, i, &type_info)
                          : api_->SessionGetOutputTypeInfo(session_, i, &type_info);
            if (!check(st, "SessionGetTypeInfo")) return false;

            const OrtTensorTypeAndShapeInfo* tinfo = nullptr;
            if (check(api_->CastTypeInfoToTensorInfo(type_info, &tinfo), "CastTypeInfoToTensorInfo")
                && tinfo != nullptr) {
                discard(api_, api_->GetTensorElementType(tinfo, &info.type),
                        "GetTensorElementType");
                size_t dims = 0;
                discard(api_, api_->GetDimensionsCount(tinfo, &dims), "GetDimensionsCount");
                info.shape.resize(dims);
                if (dims > 0) {
                    discard(api_, api_->GetDimensions(tinfo, info.shape.data(), dims),
                            "GetDimensions");
                }
            }
            api_->ReleaseTypeInfo(type_info);
            out->push_back(std::move(info));
        }
        return true;
    };

    inputs_.clear();
    outputs_.clear();
    if (!collect(true, n_in, &inputs_))   return false;
    if (!collect(false, n_out, &outputs_)) return false;
    return true;
}

bool Session::open(const json& config) {
    // ★ロード前の prereclaim（EdgeSLM_AndroidScale P0A で効果を確認、P2 の 1.5B 用に通常経路へ入れた）。
    //   ロードは数秒のうちに「binary 分のヒープ＋dma-buf＋ページキャッシュ」を要求する。MemFree が少ないと
    //   lmkd がホームやキーボードまで kill し、最後にこのアプリも kill する。先にキャッシュを回収させておく。
    //   job（またはデモの kv_modes.json の job）に prereclaim_mb があるときだけ行う。既定は何もしない
    const long long pre_mb = config.value("prereclaim_mb", 0LL);
    if (pre_mb > 0) {
        const json r = prereclaim(pre_mb, config.value("prereclaim_step_mb", 32LL),
                                  config.value("prereclaim_sleep_ms", 50));
        LOGI("prereclaim %lld MB in %.0f ms: MemFree %lld kB -> %lld kB", r.value("allocated_mb", 0LL),
             r.value("alloc_ms", 0.0), r["before"].value("mem_free_kb", -1LL),
             r["after"].value("mem_free_kb", -1LL));
    }
    const json runtime = config.value("runtime", json::object());
    const std::string native_lib_dir = runtime.value("native_lib_dir", std::string());

    // ★既定では FastRPC のパスを自分で設定しない。ORT の QNN EP に任せる。
    // 先に設定すると ORT が自分の設定をスキップしてデバイス生成に失敗する
    // （configure_fastrpc_paths の説明を参照）。
    if (runtime.value("set_fastrpc_paths", false)) {
        LOGI("runtime.set_fastrpc_paths=true: setting paths ourselves (ORT will skip its own)");
        configure_fastrpc_paths(native_lib_dir);
    } else {
        LOGI("runtime.set_fastrpc_paths=false: leaving ADSP_LIBRARY_PATH to ORT");
    }

    if (!load_ort()) return false;

    if (!env_) {
        if (!check(api_->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "edgeslm", &env_), "CreateEnv"))
            return false;
    }

    config_ = config;

    if (!check(api_->CreateRunOptions(&run_options_), "CreateRunOptions")) return false;
    if (!check(api_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info_),
               "CreateCpuMemoryInfo")) return false;

    return reopen(config.value("model", std::string()));
}

bool Session::reopen(const std::string& model_rel) {
    if (!api_) {
        set_error("reopen", "session was never opened");
        return false;
    }
    // モデルを張り替えたら **KVキャッシュの持ち越しは無効**。
    // 形も意味も変わるので、残っていると次の generate_kv が静かに壊れる。
    kv_carry_.reset();
    const std::string base_dir = config_.value("runtime", json::object())
                                        .value("base_dir", std::string());
    const std::string path = resolve(base_dir, model_rel);
    if (path.empty()) {
        set_error("reopen", "model path is empty");
        return false;
    }

    // 前のセッションを閉じる。Env / RunOptions / MemoryInfo は使い回す。
    free_shared();                            // 共有メモリは前のセッションのアロケータのもの
    if (session_) {
        api_->ReleaseSession(session_);
        session_ = nullptr;
    }
    inputs_.clear();
    outputs_.clear();
    backend_label_.clear();
    strict_succeeded_ = false;
    ctx_retries_ = 0;
    model_path_ = path;
    LOGI("model=%s", model_path_.c_str());

    // ① まず strict（CPU フォールバック禁止）で開く。
    // ② 失敗したら通常で開き直す。どちらで開けたかがグラフ配置のラベルになる。
    strict_succeeded_ = create_session(model_path_, config_, /*strict=*/true);
    // ★context の生成が error 1002 で失敗したときは、**strict のまま**少し待って開き直す。
    //   直前に別のモデルを閉じた直後だと一時的に起きる（2026-09-19 EdgeSLM_AndroidScale Phase 3:
    //   デモの KV 画面で Qwen3 → 1.5B に切り替えたとき strict が 1002、3.3秒後の②は通った。P0A e7 でも1回）。
    //   以前はそのまま②へ進み、EPContext だけのモデルでも「一部 CPU」と表示されていた
    //   （EPContext のモデルは②でも配置は同じ。②は実質ただの再試行だった）。
    //   上限超え（1プロセス約1,710MiB）のような恒常的な 1002 なら、再試行しても同じく失敗するだけ。
    const int ctx_retry_max = config_.value("ctx_retry_max", 2);
    const int ctx_retry_wait_ms = config_.value("ctx_retry_wait_ms", 1500);
    while (!strict_succeeded_ && ctx_retries_ < ctx_retry_max &&
           last_error_.find("Error code: 1002") != std::string::npos) {
        ++ctx_retries_;
        LOGI("context create failed with 1002; retry strict in %d ms (%d/%d)", ctx_retry_wait_ms,
             ctx_retries_, ctx_retry_max);
        std::this_thread::sleep_for(std::chrono::milliseconds(ctx_retry_wait_ms));
        strict_succeeded_ = create_session(model_path_, config_, /*strict=*/true);
    }
    if (!strict_succeeded_) {
        if (!create_session(model_path_, config_, /*strict=*/false)) {
            return false;  // どちらでも開けなかった
        }
    }
    backend_label_ = strict_succeeded_ ? "QNN HTP - EPContext - all nodes"
                                       : "QNN HTP - EPContext - CPU I/O boundary";
    LOGI("session opened: %s", backend_label_.c_str());

    // ★HTP 共有メモリのアロケータ（Phase 7）。取れなければ従来どおり（ヒープのバッファ）で動く
    if (config_.value("htp_shared_io", false)) {
        OrtMemoryInfo* mi = nullptr;
        if (check(api_->CreateMemoryInfo("QnnHtpShared", OrtDeviceAllocator, 0, OrtMemTypeDefault, &mi),
                  "CreateMemoryInfo(QnnHtpShared)")) {
            if (check(api_->CreateAllocator(session_, mi, &shared_alloc_), "CreateAllocator(QnnHtpShared)")) {
                discard(api_, api_->AllocatorGetInfo(shared_alloc_, &shared_mem_info_), "AllocatorGetInfo");
                LOGI("HTP shared memory allocator: ready");
            } else {
                shared_alloc_ = nullptr;
            }
            api_->ReleaseMemoryInfo(mi);
        }
    }

    return collect_io();
}

bool Session::run(const std::vector<uint8_t>& input_bytes, double* out_ms) {
    return run(std::vector<std::vector<uint8_t>>{input_bytes}, out_ms);
}

std::string Session::resolve_path(const std::string& p) const {
    const std::string base_dir = config_.value("runtime", json::object())
                                        .value("base_dir", std::string());
    return resolve(base_dir, p);
}

bool Session::run(const std::vector<std::vector<uint8_t>>& inputs, double* out_ms) {
    return run(inputs, out_ms, nullptr);
}

bool Session::run(const std::vector<std::vector<uint8_t>>& inputs, double* out_ms,
                  std::vector<std::vector<uint8_t>>* outputs) {
    if (!session_ || inputs_.empty() || outputs_.empty()) {
        set_error("run", "session is not ready");
        return false;
    }
    // 数が合わないまま黙って先頭だけ渡すと、原因の分かりにくい失敗になる。
    if (inputs.size() != inputs_.size()) {
        set_error("run", "input count mismatch: model expects " +
                             std::to_string(inputs_.size()) + ", got " +
                             std::to_string(inputs.size()));
        return false;
    }

    std::vector<const char*> in_names;
    std::vector<const char*> out_names;
    in_names.reserve(inputs_.size());
    out_names.reserve(outputs_.size());
    for (const auto& i : inputs_)  in_names.push_back(i.name.c_str());
    for (const auto& o : outputs_) out_names.push_back(o.name.c_str());

    std::vector<OrtValue*> in_values(inputs_.size(), nullptr);
    bool built = true;
    for (size_t i = 0; i < inputs_.size(); ++i) {
        const TensorInfo& t = inputs_[i];
        OrtStatus* st = api_->CreateTensorWithDataAsOrtValue(
            mem_info_,
            const_cast<uint8_t*>(inputs[i].data()), inputs[i].size(),
            t.shape.data(), t.shape.size(), t.type, &in_values[i]);
        if (!check(st, "CreateTensorWithDataAsOrtValue(" + t.name + ")")) {
            built = false;
            break;
        }
    }
    if (!built) {
        for (OrtValue* v : in_values) if (v) api_->ReleaseValue(v);
        return false;
    }

    std::vector<OrtValue*> out_values(outputs_.size(), nullptr);

    const double t0 = now_ms();
    OrtStatus* st = api_->Run(session_, run_options_,
                              in_names.data(),
                              (const OrtValue* const*)in_values.data(), in_values.size(),
                              out_names.data(), out_names.size(), out_values.data());
    const double t1 = now_ms();

    const bool ok = check(st, "Run");

    if (ok && outputs != nullptr) {
        outputs->clear();
        outputs->reserve(out_values.size());
        for (size_t i = 0; i < out_values.size(); ++i) {
            std::vector<uint8_t> buf;
            OrtTensorTypeAndShapeInfo* info = nullptr;
            if (out_values[i] != nullptr &&
                check(api_->GetTensorTypeAndShape(out_values[i], &info), "GetTensorTypeAndShape")) {
                size_t count = 0;
                ONNXTensorElementDataType t = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
                discard(api_, api_->GetTensorShapeElementCount(info, &count),
                        "GetTensorShapeElementCount");
                discard(api_, api_->GetTensorElementType(info, &t), "GetTensorElementType(out)");
                TensorInfo ti;
                ti.type = t;
                const size_t esz = ti.element_size();
                void* data = nullptr;
                if (esz > 0 && count > 0 &&
                    check(api_->GetTensorMutableData(out_values[i], &data), "GetTensorMutableData")
                    && data != nullptr) {
                    buf.assign(static_cast<uint8_t*>(data),
                               static_cast<uint8_t*>(data) + count * esz);
                }
                api_->ReleaseTensorTypeAndShapeInfo(info);
            }
            outputs->push_back(std::move(buf));
        }
    }

    for (OrtValue* v : out_values) if (v) api_->ReleaseValue(v);
    for (OrtValue* v : in_values)  if (v) api_->ReleaseValue(v);

    if (ok && out_ms) *out_ms = t1 - t0;
    return ok;
}

bool Session::run_into(const std::vector<const void*>& in_ptrs, const std::vector<void*>& out_ptrs,
                       double* out_ms, bool shared) {
    if (shared && !shared_mem_info_) {
        set_error("run_into", "shared memory is not enabled");
        return false;
    }
    const OrtMemoryInfo* mi = shared ? shared_mem_info_ : mem_info_;
    if (!session_ || in_ptrs.size() != inputs_.size() || out_ptrs.size() != outputs_.size()) {
        set_error("run_into", "session is not ready, or pointer count mismatch");
        return false;
    }
    std::vector<const char*> in_names;
    std::vector<const char*> out_names;
    for (const auto& i : inputs_)  in_names.push_back(i.name.c_str());
    for (const auto& o : outputs_) out_names.push_back(o.name.c_str());
    std::vector<OrtValue*> in_values(inputs_.size(), nullptr);
    std::vector<OrtValue*> out_values(outputs_.size(), nullptr);
    auto wrap = [&](const TensorInfo& t, void* p, OrtValue** v) {
        return check(api_->CreateTensorWithDataAsOrtValue(
                         mi, p, t.element_count() * t.element_size(),
                         t.shape.data(), t.shape.size(), t.type, v),
                     "CreateTensorWithDataAsOrtValue(" + t.name + ")");
    };
    bool ok = true;
    for (size_t i = 0; ok && i < inputs_.size(); ++i)
        ok = wrap(inputs_[i], const_cast<void*>(in_ptrs[i]), &in_values[i]);
    // ★出力も呼び出し側のバッファで包んで渡す。ORT はそこへ直接書く（確保しない）
    for (size_t i = 0; ok && i < outputs_.size(); ++i)
        ok = wrap(outputs_[i], out_ptrs[i], &out_values[i]);
    if (ok) {
        const double t0 = now_ms();
        OrtStatus* st = api_->Run(session_, run_options_,
                                  in_names.data(), (const OrtValue* const*)in_values.data(), in_values.size(),
                                  out_names.data(), out_names.size(), out_values.data());
        const double t1 = now_ms();
        ok = check(st, "Run");
        if (ok && out_ms) *out_ms = t1 - t0;
    }
    for (OrtValue* v : out_values) if (v) api_->ReleaseValue(v);
    for (OrtValue* v : in_values)  if (v) api_->ReleaseValue(v);
    return ok;
}

uint8_t* Session::shared_buf(const std::string& key, size_t bytes) {
    if (!shared_alloc_) return nullptr;
    auto it = shared_bufs_.find(key);
    if (it != shared_bufs_.end()) {
        if (it->second.second == bytes) return static_cast<uint8_t*>(it->second.first);
        api_->AllocatorFree(shared_alloc_, it->second.first);
        shared_bufs_.erase(it);
    }
    void* p = nullptr;
    if (!check(api_->AllocatorAlloc(shared_alloc_, bytes, &p), "AllocatorAlloc(QnnHtpShared)") || !p) {
        return nullptr;
    }
    std::memset(p, 0, bytes);
    shared_bufs_[key] = {p, bytes};
    return static_cast<uint8_t*>(p);
}

void Session::free_shared() {
    if (!api_) return;
    if (shared_alloc_) {
        for (auto& kv : shared_bufs_) api_->AllocatorFree(shared_alloc_, kv.second.first);
        api_->ReleaseAllocator(shared_alloc_);
    }
    shared_bufs_.clear();
    shared_alloc_ = nullptr;
    shared_mem_info_ = nullptr;
}

json Session::describe() const {
    json j;
    j["ok"] = (session_ != nullptr);
    j["backend"] = backend_label_;
    j["strict_no_cpu_fallback"] = strict_succeeded_;
    j["ctx_retries"] = ctx_retries_;          // error 1002 で strict を開き直した回数
    j["htp_shared_io"] = shared_alloc_ != nullptr;
    {
        size_t tot = 0;
        for (const auto& kv : shared_bufs_) tot += kv.second.second;
        j["shared_bytes"] = tot;
    }
    j["model"] = model_path_;
    if (!last_error_.empty()) j["last_error"] = last_error_;

    auto dump_io = [](const std::vector<TensorInfo>& v) {
        json arr = json::array();
        for (const auto& t : v) {
            json e;
            e["name"] = t.name;
            e["type"] = type_name(t.type);
            e["shape"] = t.shape;
            e["static"] = (t.element_count() != 0);
            arr.push_back(e);
        }
        return arr;
    };
    j["inputs"] = dump_io(inputs_);
    j["outputs"] = dump_io(outputs_);
    return j;
}

json Session::versions() {
    json j;
    void* lib = dlopen("libonnxruntime.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        j["onnxruntime"] = "unavailable";
        j["error"] = dlerror() ? dlerror() : "(null)";
        return j;
    }
    using GetApiBaseFn = const OrtApiBase* (*)(void);
    auto get_api_base = reinterpret_cast<GetApiBaseFn>(dlsym(lib, "OrtGetApiBase"));
    if (get_api_base) {
        const OrtApiBase* base = get_api_base();
        j["onnxruntime"] = base->GetVersionString() ? base->GetVersionString() : "unknown";
        j["ort_api_version"] = ORT_API_VERSION;
    } else {
        j["onnxruntime"] = "unknown";
    }
    // QNN の版はランタイム API から直接は取れないため、
    // 同梱した QAIRT のセット版をホスト側 manifest と突き合わせる（FR-12）。
    j["qnn_runtime"] = "see manifest (bundled QAIRT set)";
    return j;
}

}  // namespace edgeslm
