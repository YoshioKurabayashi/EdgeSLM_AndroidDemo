package org.edgeslm.demo

/**
 * JNI 境界。
 *
 * 設計方針（基本設計③ 6.2節）:
 *   引数・戻り値を **JSON 文字列に統一**している。パラメータを1つ増やすたびに
 *   シグネチャが変わると、そのたびに APK の再ビルドが必要になるため。
 *   JSON なら変更が C++ 側に閉じ、この Kotlin ファイルは触らずに済む。
 *
 *   ビルドは（JVM の通信制約により）ユーザー実行になるため、
 *   再ビルドを要する変更を減らすことが設計上の最優先事項である。
 */
object NativeBridge {

    init {
        System.loadLibrary("edgeslm_runner")
    }

    /**
     * ORT / QNN のセッションを生成する。
     * @param configJson モデルパス・EP オプション・セッション設定を含む JSON
     * @return セッションハンドル。0 なら失敗（理由は describe/エラー JSON で取得）
     */
    external fun nativeCreate(configJson: String): Long

    /**
     * バックエンド名・グラフ配置・入出力情報を JSON で返す。
     * セッション生成に失敗した場合もエラー内容を JSON で返す。
     */
    external fun nativeDescribe(handle: Long): String

    /**
     * ジョブを実行し、結果を JSON で返す。
     * task（probe / bench / op_check / soak / generate）は jobJson の中で指定する。
     */
    external fun nativeRunJob(handle: Long, jobJson: String): String

    /**
     * 生成の途中経過を JSON で返す（task=generate 実行中に別スレッドから読む）。
     * nativeRunJob は生成が終わるまで戻らないため、逐次表示はこちらで行う。
     * 返す JSON: running / done / ids[] / n / last_ms / median_ms / median_ort_ms /
     *            tokens_per_sec / rev（前回から進んだかの判定用）
     */
    external fun nativeProgress(): String

    /** soak / generate の中断要求 */
    external fun nativeCancel(handle: Long): Boolean

    /** セッションを破棄する */
    external fun nativeDestroy(handle: Long)

    /**
     * 1プロセスに NPU のモデルを順に開いて保持し、メモリの推移を記録する（EdgeSLM_AndroidScale Phase 0-A）。
     * **ハンドルを取らない。** モデルを開く前の状態から測るため、開閉は C++ 側が自分で行う。
     * 途中経過は out/memprobe_<job_id>_steps.jsonl / _samples.csv に逐次書かれる。
     */
    external fun nativeMemProbe(jobJson: String): String

    /** ORT / QNN の版を JSON で返す（FR-12 の記録用。セッション不要） */
    external fun nativeVersions(): String
}
