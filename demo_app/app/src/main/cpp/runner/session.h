// EdgeSLM Runner - ORT/QNN セッション管理
//
// 設計上の要点（基本設計③ / 参考実装_ZipDepth解説 5章）:
//  - ORT は dlopen("libonnxruntime.so") + dlsym("OrtGetApiBase") で動的に読む
//  - QNN EP の backend_path には **実ファイルパス**（nativeLibraryDir/libQnnHtp.so）を渡す。
//    FastRPC が V68 skeleton に通常のファイルシステムパスを要求するため（C-6）
//  - グラフ配置のラベルは推測しない。**disable_cpu_ep_fallback=1 で成功したか**で決める。
//    strict 成功 → "all nodes" / strict 失敗かつ通常成功 → "CPU I/O boundary"
//  - 電力プロファイル（htp_performance_mode / rpc_control_latency）は **設定しない**。
//    unsigned PD では setPowerConfig が許可されず Run が失敗する（要件定義書 3.4節 C-3）

#pragma once

#include <memory>
#include <map>
#include <string>
#include <vector>

#include "ort/onnxruntime_c_api.h"
#include "third_party/json.hpp"

namespace edgeslm {

struct TensorInfo {
    std::string name;
    std::vector<int64_t> shape;   // 動的次元は -1
    ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    size_t element_count() const;                 // 動的次元がある場合は 0
    size_t element_size() const;                  // 型1要素のバイト数
};

/// 会話をまたいで KVキャッシュを持ち回すための状態。
///
/// ★**Session が持つ。** 最初 `static std::map<const Session*, KvCarry>` にしたが、
///   セッションを作り直すと**解放済みアドレスが再利用され、別セッションの状態を拾う**
///   （2026-09-14 実機で発生。4往復目だけ再利用が効いたように見えた）。
///   KVキャッシュはそのセッションのものなので、寿命もセッションに合わせる。
struct KvCarry {
    std::string model;
    std::vector<std::string> cache_names;        // "past_k_0" など
    std::vector<std::vector<uint8_t>> caches;    // 対応する中身
    int64_t n_valid = 0;                         // マスクで開けてよいスロット数
    int64_t pos = 0;                             // 次に使う RoPE の絶対位置
    std::vector<int64_t> tokens;                 // これまで食わせたトークン列
    // 位置0 の K/V（past_* の入力順に [KVH][HD] float32）。`pos0_kv` のファイルを持たないモデルで
    // `pin_pos0` を使うとき、1ターン目の最初のステップで端末が計算したものを取っておく（2026-09-19）
    std::vector<uint8_t> pos0;
    bool valid = false;

    void reset() { *this = KvCarry(); }

    /// `ids` のうち**何トークンまで再利用できるか**。前方一致しなければ 0。
    ///
    /// ★トークン列を保持して突き合わせるのが肝。クライアントが履歴を編集したり
    ///   別の会話を始めたりしたときに、**黙って壊れたキャッシュを使い続けない**。
    size_t reusable(const std::vector<int64_t>& ids) const {
        if (!valid || tokens.empty() || tokens.size() > ids.size()) return 0;
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (tokens[i] != ids[i]) return 0;
        }
        return tokens.size();
    }
};

class Session {
public:
    Session();
    ~Session();

    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    // config は job.json 全体。runtime.native_lib_dir / runtime.base_dir を含む前提。
    bool open(const nlohmann::json& config);

    // 同じ Session で**別のモデルに張り替える**（v0.2）。
    // op_check のように多数のモデルを次々試す用途のために足した。
    // OrtEnv を作り直さないのは、QNN バックエンドがプロセス全体の状態を持つためで、
    // モデルごとに Env を作って壊すと不安定になる（Session のデストラクタの注記と同じ理由）。
    bool reopen(const std::string& model_rel);

    // 1回の推論。入力バッファは呼び出し側が用意する。所要時間(ms)を out_ms に返す。
    // 多入力版が本体で、単入力版はその薄い包み（KVキャッシュは多入力になる）。
    // outputs を渡すと出力テンソルの中身を取り出す（v0.3）。
    // **実機での精度を測るために足した**——ここまでの精度はすべてホスト上の
    // シミュレーションで、HTP の演算順序・丸めの違いが含まれていなかった。
    bool run(const std::vector<std::vector<uint8_t>>& inputs, double* out_ms,
             std::vector<std::vector<uint8_t>>* outputs);
    bool run(const std::vector<std::vector<uint8_t>>& inputs, double* out_ms);
    bool run(const std::vector<uint8_t>& input_bytes, double* out_ms);

    // ★コピー無しの1回の推論（EdgeSLM_AndroidScale Phase 6）。入力も出力も**呼び出し側のバッファを直接使う**。
    //   上の run は出力を ORT に毎回確保させ（1.5B では 57本・約15MB）、それを vector へ写していた。
    //   in_ptrs / out_ptrs は inputs() / outputs() の順。大きさは TensorInfo の形と型から決まる（静的な形のみ）。
    bool run_into(const std::vector<const void*>& in_ptrs, const std::vector<void*>& out_ptrs,
                  double* out_ms, bool shared = false);

    // ★HTP 共有メモリ（Phase 7）。job の `htp_shared_io: true` で QNN EP の
    //   `enable_htp_shared_memory_allocator` を有効にし、NPU と共有するメモリ（rpcmem）を確保できるようにする。
    //   このメモリで包んだテンソルは、QNN EP が NPU 側へ**登録して渡す**（コピーしない。登録は番地ごとに1回）。
    bool shared_io_enabled() const { return shared_alloc_ != nullptr; }
    /// key ごとに1つ、共有メモリのバッファを返す（**セッションと同じ寿命**。大きさが違えば取り直す）。無効なら nullptr
    uint8_t* shared_buf(const std::string& key, size_t bytes);

    // job.json の相対パスを base_dir から解決する（infer task のファイル入出力用）
    std::string resolve_path(const std::string& p) const;

    const std::vector<TensorInfo>& inputs() const { return inputs_; }
    const std::vector<TensorInfo>& outputs() const { return outputs_; }

    nlohmann::json describe() const;

    bool ok() const { return session_ != nullptr; }
    const std::string& backend_label() const { return backend_label_; }
    const std::string& last_error() const { return last_error_; }

    static nlohmann::json versions();

    /// KVキャッシュの持ち越し（`generate_kv` の `reuse_cache`）。
    /// **セッションと同じ寿命**。張り替え（reopen）では捨てる。
    KvCarry& kv_carry() { return kv_carry_; }

    /// `generate_kv` の zero_copy 用のバッファ（KV のもう1組と、KV 以外の出力）。**セッションと同じ寿命**。
    /// ターンごとに作り直すと、各ターンの充填のステップが約9ms 遅くなった（Phase 6。新しいバッファを
    /// NPU 側が使えるようにする準備が毎ターン走るとみている）。
    std::vector<std::vector<uint8_t>>& zc_kv_b() { return zc_kv_b_; }
    std::vector<std::vector<uint8_t>>& zc_out() { return zc_out_; }

private:
    bool load_ort();
    bool create_session(const std::string& model_path,
                        const nlohmann::json& config,
                        bool strict);
    bool collect_io();
    void set_error(const std::string& where, const std::string& what);
    bool check(OrtStatus* status, const std::string& where);

    void* ort_lib_ = nullptr;
    const OrtApi* api_ = nullptr;
    OrtEnv* env_ = nullptr;
    OrtSession* session_ = nullptr;
    OrtRunOptions* run_options_ = nullptr;
    OrtMemoryInfo* mem_info_ = nullptr;

    std::vector<TensorInfo> inputs_;
    std::vector<TensorInfo> outputs_;

    std::string backend_label_;
    std::string last_error_;
    std::string model_path_;
    bool strict_succeeded_ = false;
    int ctx_retries_ = 0;                     // error 1002 で strict を開き直した回数（open ごと）
    // ---- HTP 共有メモリ（Phase 7）----
    OrtAllocator* shared_alloc_ = nullptr;             // QNN EP の HtpSharedMemoryAllocator
    const OrtMemoryInfo* shared_mem_info_ = nullptr;   // shared_alloc_ が持つ（解放しない）
    std::map<std::string, std::pair<void*, size_t>> shared_bufs_;
    void free_shared();                                // セッションを閉じる前に必ず呼ぶ
    nlohmann::json config_;   // reopen で EP オプションを再利用するため保持する
    KvCarry kv_carry_;        // 会話をまたぐ KVキャッシュ（generate_kv）
    std::vector<std::vector<uint8_t>> zc_kv_b_;   // zero_copy: KV のもう1組
    std::vector<std::vector<uint8_t>> zc_out_;    // zero_copy: KV 以外の出力（logits）
};

}  // namespace edgeslm
