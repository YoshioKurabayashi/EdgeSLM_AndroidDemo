package org.edgeslm.demo

import android.app.Activity
import android.content.Intent
import android.graphics.Color
import android.graphics.Typeface
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.text.style.StyleSpan
import android.util.Log
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.Button
import android.widget.EditText
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.concurrent.thread

/**
 * KVキャッシュ版のデモ画面。
 *
 * ## なぜ別画面にしたか
 *
 * 窓ずらし版（[DemoActivity]）は**既に検証を通した経路**である。
 * そこへ手を入れずに並べたいので、**Activity ごと分けた**。
 * `DemoActivity.kt` はこの追加で1文字も変わっていない。
 *
 * ネイティブ側も `generate`（窓ずらし）とは別の task（`generate_kv`）を使う。
 *
 * ## 窓ずらしとの違い（この画面で見せたいこと）
 *
 * | | 窓ずらし | KVキャッシュ decode |
 * |---|---|---|
 * | 1ステップで入れるトークン | 窓のぶん全部（16 or 64） | **1つだけ** |
 * | 過去の文脈 | 毎回入れ直して計算し直す | **キャッシュで持ち回す** |
 * | 実測（24層） | 36.7ms（窓16）/ 42.0ms（窓64） | **28.5ms** |
 * | 生成前の準備 | 不要 | **キャッシュ充填**（プロンプトを1トークンずつ） |
 *
 * 充填は生成前に1回だけ走る。プロンプト14トークンで実測 474ms だった。
 * **この時間は生成トークンあたりの時間には含めない**（含めると速く見えすぎる）ので、
 * 画面でも分けて出す。
 *
 * ## 同時には動かせない
 *
 * セッションは**同時に2つ持たない**（2026-09-19 実測: 772MB×2 は保持できるが、ロード中の一時的なメモリ要求で lmkd に kill されうる。1プロセスの NPU 側上限も約1,710MiB で、Qwen2.5 KV512 と Qwen3 を同時に持つと超える。EdgeSLM_AndroidScale P0A）。
 * この画面を開くと窓ずらし版のセッションは閉じる。逆も同じ。
 */
class KvDemoActivity : Activity() {

    private lateinit var statusView: TextView
    private lateinit var promptEdit: EditText
    private lateinit var outputView: TextView
    private lateinit var statsView: TextView
    private lateinit var generateButton: Button
    private lateinit var stopButton: Button
    private lateinit var outputScroll: ScrollView

    @Volatile private var handle: Long = 0L
    @Volatile private var tokenizer: Qwen2Tokenizer? = null
    @Volatile private var generating = false

    private var promptText: String = ""
    private var lastRev: Long = -1L
    private var cacheLen: Int = 0

    // ---- 対話モード ----
    //
    // ★既存の「続きを書かせる」動作は**そのまま残す**。チェックを入れたときだけ
    //   会話として振る舞う。検証済みの経路へ手を入れない、というこれまでの方針と同じ。
    //
    // 会話では **KVキャッシュを持ち回す**（`reuse_cache`）。毎ターン全履歴を
    // 入れ直すと充填時間が履歴の長さに比例して伸びる（実測: 1往復 924ms → 4往復 5550ms）。
    // 持ち回せば**差分ぶんだけ**で済む（実測: 5往復 132トークンでも 816ms）。
    private var chatMode = false
    private var chatTurns = 0                    // 何往復したか
    private val chatLog = StringBuilder()        // 画面に出す会話
    private var chatStarted = false              // 1ターン目かどうか
    // 停止トークンで終わったか（記録用）。差分の組み立てには使わない——
    // その `<|im_end|>` はキャッシュに入っていないため（`chatDelta` の注記）。
    private var lastTurnEndedWithEos = false
    private val presetButtons = ArrayList<Button>()

    /// プリセットの見出しを、いまのモードに合わせる。
    private fun refreshPresets() {
        val list = if (chatMode) CHAT_PRESETS else PRESETS
        presetButtons.forEachIndexed { i, b -> b.text = list[i].first }
    }

    // モード（重みの量子化の版など）。`files/jobs/kv_modes.json` があるときだけ出る。
    private var modes: List<Pair<String, JSONObject>> = emptyList()
    @Volatile private var modeIndex = 0
    private val ui = Handler(Looper.getMainLooper())

    // ---- テスト用のインテント引数（EdgeSLM_AndroidScale Phase 3。**渡さなければ従来どおり**）----
    //   am start -n org.edgeslm.demo/.KvDemoActivity --ei mode 2 --ez chat true \
    //            --es prompts "エアコンを22度にして|音楽をかけて|さっき何度にしたっけ？"
    //   mode    : 起動時に開くモード（kv_modes.json の添字）
    //   chat    : 対話モードで始める
    //   prompts : 各ターンが終わるたびに次の文を自動で送る（`|` 区切り）。終わったら会話を
    //             out/kv_ui_chat.txt に書き、logcat に `auto_chat done` を出す
    // 1.5B の画面への組み込みとロードの安定性を、PC から繰り返し確かめるために足した。
    private val autoPrompts = ArrayDeque<String>()
    private lateinit var chatToggle: android.widget.CheckBox
    private lateinit var titleView: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        modes = loadModes()
        val m = intent?.getIntExtra("mode", -1) ?: -1
        if (m in modes.indices) modeIndex = m
        setContentView(buildUi())
        if (intent?.getBooleanExtra("chat", false) == true) chatToggle.isChecked = true
        intent?.getStringExtra("prompts")?.split("|")?.map { it.trim() }?.filter { it.isNotEmpty() }
            ?.let { autoPrompts.addAll(it) }
        setBusy(true, "起動中…")
        thread(name = "edgeslm-kv-init") { initSession() }
    }

    /// テスト用: 次の文を自動で送る。無くなったら会話を書き出す。
    private fun sendNextAuto() {
        if (generating || handle == 0L) return
        val next = autoPrompts.removeFirstOrNull() ?: return
        promptEdit.setText(next)
        onGenerate()
    }

    private fun dumpAutoChat() {
        runCatching {
            File(getExternalFilesDir(null), "out/kv_ui_chat.txt").writeText(chatLog.toString())
        }
        Log.i(TAG, "auto_chat done turns=$chatTurns")
    }

    override fun onDestroy() {
        if (handle != 0L) {
            NativeBridge.nativeCancel(handle)
            NativeBridge.nativeDestroy(handle)
            handle = 0L
        }
        super.onDestroy()
    }

    // ------------------------------------------------------------------ 画面

    private fun buildUi(): View {
        val dp = resources.displayMetrics.density
        fun px(v: Int) = (v * dp).toInt()

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(px(16), px(16), px(16), px(16))
            setBackgroundColor(Color.WHITE)
        }

        titleView = TextView(this).apply {
            text = titleFor(modeIndex)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 17f)
            setTypeface(typeface, Typeface.BOLD)
            setTextColor(Color.BLACK)
        }
        root.addView(titleView)
        root.addView(TextView(this).apply {
            text = "1トークンずつ入れて、過去は KVキャッシュで持ち回す方式"
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 11f)
            setTextColor(Color.parseColor("#888888"))
        })

        statusView = TextView(this).apply {
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
            setTextColor(Color.parseColor("#666666"))
            setPadding(0, px(4), 0, px(10))
        }
        root.addView(statusView)

        // モード行。**modes が空なら何も足さない**ので、設定ファイルが無ければ従来と同じ画面。
        if (modes.size > 1) {
            val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
            modes.forEachIndexed { i, m ->
                row.addView(Button(this).apply {
                    text = m.first
                    setTextSize(TypedValue.COMPLEX_UNIT_SP, 11f)
                    setOnClickListener { switchMode(i) }
                }, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
            }
            root.addView(row, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        }

        promptEdit = EditText(this).apply {
            hint = "続きを書かせたい文章を入力"
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 15f)
            setTextColor(Color.BLACK)
            minLines = 2
            maxLines = 4
        }
        root.addView(promptEdit, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))

        // プリセット。**対話モードでは中身を入れ替える**（続きを書かせる文 → 話しかける文）。
        val presets = HorizontalScrollView(this).apply { isHorizontalScrollBarEnabled = false }
        val presetRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        for (i in PRESETS.indices) {
            presetRow.addView(Button(this).apply {
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 11f)
                setOnClickListener {
                    val list = if (chatMode) CHAT_PRESETS else PRESETS
                    promptEdit.setText(list[i].second)
                }
            })
            presetButtons.add(presetRow.getChildAt(i) as Button)
        }
        presets.addView(presetRow)
        root.addView(presets)
        refreshPresets()

        // 対話モードの切り替え
        chatToggle = android.widget.CheckBox(this).apply {
            text = "対話モード（KVキャッシュを持ち回す）"
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
            setTextColor(Color.parseColor("#444444"))
            setOnCheckedChangeListener { _, on ->
                chatMode = on
                resetChat()
                refreshPresets()
                promptEdit.hint = if (on) "話しかける（例: エアコンを22度にして）"
                                  else "続きを書かせたい文章を入力"
                generateButton.text = if (on) "送信" else "Generate"
            }
        }
        root.addView(chatToggle)

        val buttonRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        generateButton = Button(this).apply {
            text = "Generate"
            setOnClickListener { onGenerate() }
        }
        stopButton = Button(this).apply {
            text = "Stop"
            isEnabled = false
            setOnClickListener { NativeBridge.nativeCancel(handle) }
        }
        buttonRow.addView(generateButton, LinearLayout.LayoutParams(0, WRAP_CONTENT, 2f))
        buttonRow.addView(stopButton, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
        root.addView(buttonRow, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))

        root.addView(Button(this).apply {
            text = "◀ 窓ずらし版へ切り替え"
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
            setOnClickListener { switchTo(DemoActivity::class.java) }
        }, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))

        outputView = TextView(this).apply {
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 17f)
            setTextColor(Color.BLACK)
            setLineSpacing(0f, 1.25f)
            setPadding(px(14), px(14), px(14), px(14))
            setTextIsSelectable(true)
        }
        outputScroll = ScrollView(this).apply {
            setBackgroundColor(Color.parseColor("#F4F4F6"))
            addView(outputView)
        }
        // ★統計は**会話ログの上**に置く。下に置くと、会話が伸びたときに
        //   画面外へ押し出されて見えなくなる（2026-09-14 に実機で確認）。
        statsView = TextView(this).apply {
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
            setTextColor(Color.parseColor("#444444"))
            gravity = Gravity.END
            setPadding(0, px(8), 0, 0)
            setTypeface(Typeface.MONOSPACE)
        }
        root.addView(statsView)

        root.addView(outputScroll, LinearLayout.LayoutParams(MATCH_PARENT, 0, 1f).apply {
            topMargin = px(6)
        })

        return root
    }

    private fun setBusy(busy: Boolean, status: String?) {
        generateButton.isEnabled = !busy && handle != 0L && tokenizer != null
        stopButton.isEnabled = busy && generating
        status?.let { statusView.text = it }
    }

    /**
     * もう一方の画面へ切り替える。
     *
     * ★**セッションを先に解放してから** `startActivity` する。
     *   2つ同時に持たない（2026-09-19 実測: 772MB×2 は保持できるが、ロード中の一時的なメモリ要求で lmkd に kill されうる。1プロセスの NPU 側上限も約1,710MiB で、Qwen2.5 KV512 と Qwen3 を同時に持つと超える。EdgeSLM_AndroidScale P0A）。
     *   `startActivity` → `finish()` の順にすると、新しい画面の onCreate が
     *   こちらの onDestroy より先に走りうるので、順序を自分で決める。
     */
    private fun switchTo(cls: Class<*>) {
        if (generating) return
        generateButton.isEnabled = false
        statusView.text = "切り替え中…（モデルを開き直します）"
        if (handle != 0L) {
            NativeBridge.nativeCancel(handle)
            NativeBridge.nativeDestroy(handle)
            handle = 0L
        }
        startActivity(Intent(this, cls))
        finish()
    }


    // -------------------------------------------------------------- 初期化

    private fun initSession() {
        val base = getExternalFilesDir(null)
        if (base == null) {
            ui.post { setBusy(true, "外部ストレージが使えません") }
            return
        }
        listOf("models", "models/ops", "jobs", "inputs", "out", "tokenizer").forEach {
            File(base, it).mkdirs()
        }
        try {
            ui.post { setBusy(true, "トークナイザを読み込み中…") }
            val tok = Qwen2Tokenizer.load(File(base, "tokenizer"))

            ui.post { setBusy(true, "decode モデルを読み込み中…") }
            val config = JsonInject.addRuntimePaths(
                baseJobJson(),
                baseDir = base.absolutePath,
                nativeLibDir = applicationInfo.nativeLibraryDir,
                overridesJson = null
            )
            val (h, loadMs) = openSession(config)
            val describe = JSONObject(NativeBridge.nativeDescribe(h))
            Log.i(TAG, "describe=$describe")

            if (h == 0L) {
                Log.i(TAG, "session failed: mode=$modeIndex load_ms=$loadMs error=${describe.optString("error")}")
                // ★よくある原因を具体的に出す。デモ中に原因を探す時間は無い。
                //   1プロセスで NPU に置けるのは約1,710MiB（EdgeSLM_AndroidScale P0A）。
                //   1.5B（1,481MiB）は、他のモデルを開いたままだとこれを超える（error 1002）。
                ui.post {
                    setBusy(true, "モデルを開けませんでした。" +
                        "他のモデルを開いたままだと NPU の上限（1プロセス約1,710MiB）を超えます: " +
                        describe.optString("error"))
                }
                return
            }
            handle = h
            tokenizer = tok
            cacheLen = describe.optJSONArray("inputs")?.let { arr ->
                (0 until arr.length()).map { arr.getJSONObject(it) }
                    .firstOrNull { it.optString("name") == "attn_mask" }
                    ?.optJSONArray("shape")?.let { s -> s.getInt(s.length() - 1) }
            } ?: 0

            val backend = describe.optString("backend", "?")
            val strict = describe.optBoolean("strict_no_cpu_fallback", false)
            // ★PC から読むための1行（ロードの安定性を繰り返し測る。Phase 3）
            Log.i(TAG, "session ready: mode=$modeIndex load_ms=$loadMs strict=$strict backend=$backend")
            ui.post {
                statusView.text = "$backend" +
                    (if (strict) " · 全ノード NPU" else " · 一部 CPU") +
                    " · キャッシュ長 $cacheLen · 起動 ${loadMs / 1000.0}s"
                setBusy(false, null)
                if (autoPrompts.isNotEmpty()) ui.postDelayed({ sendNextAuto() }, 300)
            }
        } catch (t: Throwable) {
            Log.e(TAG, "init failed", t)
            ui.post { setBusy(true, "初期化に失敗: ${t.javaClass.simpleName}: ${t.message}") }
        }
    }

    /**
     * decode 用のジョブ。`files/jobs/kv_demo.json` があればその中身で上書きする
     * （モデルや cos/sin 表の差し替えに再ビルドを要らなくするため）。
     */
    private fun baseJobJson(): String {
        val job = JSONObject().apply {
            put("job_id", "kv_demo_ui")
            put("task", "generate_kv")
            put("model", MODEL_REL)
            put("mask_neg", -60.0)
            put("const_inputs", JSONObject().apply {
                put("cos_table", COS_TABLE)
                put("sin_table", SIN_TABLE)
            })
            put("ep", JSONObject().apply {
                put("provider", "QNN")
                put("backend", "libQnnHtp.so")
                put("options", JSONObject().apply {
                    put("offload_graph_io_quantization", "0")
                    put("qnn_context_priority", "high")
                })
                put("session", JSONObject().apply {
                    put("intra_op_num_threads", 1)
                    put("log_severity_level", 3)
                })
            })
        }
        try {
            val f = File(getExternalFilesDir(null), OVERRIDE_REL)
            if (f.isFile) {
                val ov = JSONObject(f.readText())
                for (k in ov.keys()) job.put(k, ov.get(k))
                Log.i(TAG, "job override applied from $OVERRIDE_REL")
            }
        } catch (t: Throwable) {
            Log.w(TAG, "job override ignored: ${t.javaClass.simpleName}: ${t.message}")
        }
        // 選んでいるモードを最後に重ねる（モードが無ければ何もしない）
        modes.getOrNull(modeIndex)?.second?.let { m ->
            for (k in m.keys()) job.put(k, m.get(k))
        }
        // `assistant_prefix` は UI 側だけで使う設定なので、ネイティブには渡さない
        job.remove(ASSISTANT_PREFIX_KEY)
        return job.toString()
    }

    /// 画面のタイトル。モードがあればそのラベルを出す（1.5B を足したので固定の「0.5B」をやめた）
    private fun titleFor(index: Int): String =
        modes.getOrNull(index)?.first?.let { "$it — KVキャッシュ decode" }
            ?: "Qwen2.5-0.5B — KVキャッシュ decode"

    /**
     * `files/jobs/kv_modes.json` を読む。形は `demo_modes.json` と同じ:
     *
     *     [ {"label":"per-tensor", "job":{...}}, {"label":"per-channel", "job":{...}} ]
     *
     * **無ければ空を返す**ので、設定ファイルが無い端末では従来どおり単一モードで動く。
     */
    private fun loadModes(): List<Pair<String, JSONObject>> {
        return try {
            val f = File(getExternalFilesDir(null), MODES_REL)
            if (!f.isFile) return emptyList()
            val arr = JSONArray(f.readText())
            (0 until arr.length()).map { i ->
                val o = arr.getJSONObject(i)
                o.optString("label", "mode$i") to o.getJSONObject("job")
            }
        } catch (t: Throwable) {
            Log.w(TAG, "kv_modes.json ignored: ${t.javaClass.simpleName}: ${t.message}")
            emptyList()
        }
    }

    /**
     * セッションを開き、(ハンドル, 所要ミリ秒) を返す。
     *
     * 一時的な error 1002（直前に閉じたモデルの NPU 側の後片付けが終わる前に開いたとき）は、
     * **ネイティブ側（Session::open）が strict のまま待って開き直す**（EdgeSLM_AndroidScale Phase 3）。
     * ここでは重ねて再試行しない——恒常的な失敗（NPU の上限超えなど）で利用者を長く待たせないため。
     */
    private fun openSession(config: String): Pair<Long, Long> {
        val t0 = System.currentTimeMillis()
        val h = NativeBridge.nativeCreate(config)
        return h to (System.currentTimeMillis() - t0)
    }

    /** モードを切り替える。**セッションは1つだけ開く**ので、先に解放してから開き直す。 */
    private fun switchMode(index: Int) {
        if (generating || index == modeIndex || index !in modes.indices) return
        val prev = modeIndex
        modeIndex = index
        // ★モデルを替えたら会話を最初からやり直す（2026-09-17）。
        //   セッションを開き直すので KV キャッシュは失われるのに、`chatStarted` が立ったままだと
        //   **system prompt を含まない「2ターン目の差分」**を新しいモデルへ渡してしまう
        //   （実機で promptLen=22 になり、モデルが会話の途中から始まったように振る舞った）。
        resetChat()
        setBusy(true, "${modes[index].first} に切り替え中…")
        thread(name = "edgeslm-kv-switch") {
            try {
                if (handle != 0L) {
                    NativeBridge.nativeDestroy(handle)
                    handle = 0L
                }
                val config = JsonInject.addRuntimePaths(
                    baseJobJson(),
                    baseDir = getExternalFilesDir(null)!!.absolutePath,
                    nativeLibDir = applicationInfo.nativeLibraryDir,
                    overridesJson = null
                )
                val (h, loadMs) = openSession(config)
                if (h == 0L) {
                    val err = runCatching { JSONObject(NativeBridge.nativeDescribe(0L)).optString("error") }
                        .getOrDefault("")
                    Log.i(TAG, "session failed: mode=$index load_ms=$loadMs error=$err")
                    // ★**本当に元のモードを開き直す。** 以前は表示だけ「元へ戻します」で、セッションを
                    //   開き直していなかった（画面が「処理中」のまま止まる。2026-09-19 にコードを読んで気付いた。
                    //   実機では未発生——これまで切り替えが失敗したことが無かった）。
                    modeIndex = prev
                    val back = JsonInject.addRuntimePaths(
                        baseJobJson(),
                        baseDir = getExternalFilesDir(null)!!.absolutePath,
                        nativeLibDir = applicationInfo.nativeLibraryDir,
                        overridesJson = null
                    )
                    val (hb, _) = openSession(back)
                    if (hb != 0L) {
                        handle = hb
                        ui.post {
                            titleView.text = titleFor(prev)
                            statusView.text = "${modes[index].first} を開けなかったので " +
                                "${modes[prev].first} に戻しました: $err"
                            setBusy(false, null)
                        }
                    } else {
                        ui.post { setBusy(true, "切り替えに失敗し、元のモードも開けませんでした: $err") }
                    }
                    return@thread
                }
                handle = h
                val d = JSONObject(NativeBridge.nativeDescribe(h))
                val strict = d.optBoolean("strict_no_cpu_fallback", false)
                Log.i(TAG, "mode switched: mode=$index load_ms=$loadMs strict=$strict")
                ui.post {
                    titleView.text = titleFor(index)
                    statusView.text = "${modes[index].first} · ${d.optString("backend", "?")}" +
                        (if (strict) " · 全ノード NPU" else " · 一部 CPU") +
                        " · 切替 ${loadMs / 1000.0}s"
                    setBusy(false, null)
                }
            } catch (t: Throwable) {
                modeIndex = prev
                Log.e(TAG, "switchMode failed", t)
                ui.post { setBusy(true, "切り替えに失敗: ${t.javaClass.simpleName}") }
            }
        }
    }

    // ---------------------------------------------------------------- 生成

    /// 会話を最初からやり直す。**キャッシュも捨てさせる**（`reset_cache`）。
    private fun resetChat() {
        chatTurns = 0
        chatStarted = false
        chatLog.setLength(0)
        lastTurnEndedWithEos = false
        outputView.text = ""
        statsView.text = ""
    }

    /**
     * 対話モードで今回送るトークン列を作る。
     *
     * ★**1ターン目だけ全部、2ターン目以降は差分だけ**を渡す。
     *   全履歴を毎回組み立てようとすると、**モデルが実際に何を答えたかを
     *   こちら側で正確に持っていないと前方一致しない**（応答を整形したり
     *   空白を落としたりすると一致しなくなる）。
     *   実際のチャットで足すのは「直前の応答を閉じる特殊トークンと、
     *   次のユーザー発話を包むテンプレート」だけで、**応答の中身に依存しない**。
     *   ネイティブ側が実際の応答を含む履歴を持っているので、そちらに任せる。
     */
    private fun chatDelta(user: String): String =
        if (!chatStarted)
            "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n" +
            "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n" +
            assistantPrefix()
        else
            // ★`<|im_end|>` は **常に付ける**。
            //   停止トークンで終わった場合でも、その `<|im_end|>` は
            //   **キャッシュに入っていない**（生成ループは「選ぶ→次の周回で食わせる」
            //   形なので、最後に選んだ1つは食わせずに終わる）。
            //   一度「EOS なら省く」ようにしたが、それは
            //   「キャッシュに入っている」という**成り立たない前提**だった（2026-09-14）。
            "<|im_end|>" +
            "\n<|im_start|>user\n" + user +
            "<|im_end|>\n<|im_start|>assistant\n" + assistantPrefix()

    /**
     * assistant の開始直後に差し込む定型句。**モデルによって要る／要らないが変わる**。
     *
     * Qwen3 の公式テンプレートは `enable_thinking=False` のとき、
     * `<|im_start|>assistant` の直後に **空の `<think></think>` ブロック** を入れる。
     * これは「思考ブロックは無し」とモデルに伝える印で、**落とすとモデルが
     * 推論文（`<think>` で始まる長い独り言）を垂れ流す**（ホストで再現済み）。
     * Qwen2.5 にはこの仕組みが無いので既定は空。
     *
     * ★もう一つ、Qwen3 のテンプレートは **append-only ではない**。
     *   あるターンが履歴になると、このブロックはテンプレートから**消える**。
     *   つまり2ターン目以降の「公式レンダリング」は、
     *   **既にキャッシュへ入れた列の接頭辞にならない**（ホストで
     *   共通37トークン / 既投入52トークン という形で検出した）。
     *   食べたトークンは取り消せないので、ここでは**毎ターン足す**方針を採る。
     *
     * 値は `jobs/kv_modes.json`（モードごと）か `jobs/kv_demo.json`（全体）の
     * `assistant_prefix` で与える。**ネイティブには渡さない**
     * （`baseJobJson()` が取り除く）。
     */
    private fun assistantPrefix(): String {
        modes.getOrNull(modeIndex)?.second?.let { m ->
            if (m.has(ASSISTANT_PREFIX_KEY)) return m.optString(ASSISTANT_PREFIX_KEY, "")
        }
        return try {
            val f = File(getExternalFilesDir(null), OVERRIDE_REL)
            if (f.isFile) JSONObject(f.readText()).optString(ASSISTANT_PREFIX_KEY, "") else ""
        } catch (t: Throwable) {
            Log.w(TAG, "assistant_prefix ignored: ${t.javaClass.simpleName}: ${t.message}")
            ""
        }
    }

    private fun onGenerate() {
        val tok = tokenizer ?: return
        if (handle == 0L || generating) return
        val text = promptEdit.text.toString()
        if (text.isBlank()) return
        val ids = tok.encode(if (chatMode) chatDelta(text) else text)
        if (ids.isEmpty()) return
        // ★端末が**実際に食べた** ID 列を残す。個数が合っていても中身が合うとは限らない
        //   （Qwen3 の <think> 151667 / </think> 151668 が specials.tsv に無いと
        //     BPE で刻まれて別の列になる）。ホスト側の期待値と突き合わせるため。
        Log.i(TAG, "encoded ids(${ids.size})=" + ids.take(64).joinToString(","))

        if (chatMode) {
            chatLog.append("あなた: ").append(text).append("\n")
            promptEdit.setText("")
        }
        promptText = if (chatMode) chatLog.toString() else text
        lastRev = -1L
        generating = true
        setBusy(true, null)
        renderOutput(emptyList())
        // 充填中は生成が始まらないので、待たせていることを明示する
        statsView.text = "キャッシュを埋めています（${ids.size} トークン）…"

        val job = JSONObject(baseJobJson()).apply {
            // 対話は文が途中で切れると読めないので長めに取る。
            // 続きを書かせるモードは従来どおり。
            put("max_new", if (chatMode) MAX_NEW_CHAT else MAX_NEW)
            put("prompt_ids", JSONArray().also { a -> ids.forEach { a.put(it) } })
            put("stop_ids", JSONArray().also { a -> STOP_IDS.forEach { a.put(it) } })
            if (chatMode) {
                // ★2ターン目以降は前ターンのキャッシュを使う。
                //   ネイティブ側が**トークン列の前方一致**を確かめてから再利用するので、
                //   食い違っていれば安全に作り直される（黙って壊れた文脈で続かない）。
                put("reuse_cache", true)
                put("reset_cache", !chatStarted)
                // ★2ターン目以降は `prompt_ids` が**差分**なので、
                //   ネイティブ側にこれまでの履歴を前置させる。
                //   これを付け忘れると「履歴より短いプロンプト」になって前方一致せず、
                //   **文脈ゼロで生成**してしまう（2026-09-14 に実機で踏んだ）。
                put("prompt_is_delta", chatStarted)
            }
        }
        chatStarted = true
        val jobJson = JsonInject.addRuntimePaths(
            job.toString(),
            baseDir = getExternalFilesDir(null)!!.absolutePath,
            nativeLibDir = applicationInfo.nativeLibraryDir,
            overridesJson = null
        )

        ui.post(pollProgress)
        thread(name = "edgeslm-kv-generate") {
            val result = NativeBridge.nativeRunJob(handle, jobJson)
            // テスト用: 結果の JSON を丸ごと残す（logcat は1行が切れるので、1ステップごとの時間などはここから読む）
            if (intent?.hasExtra("prompts") == true) runCatching {
                File(getExternalFilesDir(null), "out/kv_turn_${chatTurns + 1}.json").writeText(result)
            }
            Log.i(TAG, "generate_kv result=${result.take(600)}")
            // ★1トークン目の top-10 / probe を**短い専用行**で出す。
            //   result 全体は describe を含むので logcat の1行上限で切れてしまう。
            runCatching {
                val o = JSONObject(result)
                if (o.has("first_topk")) Log.i(TAG, "first_topk=" + o.get("first_topk"))
                if (o.has("first_probe")) Log.i(TAG, "first_probe=" + o.get("first_probe"))
            }
            ui.post {
                generating = false
                setBusy(false, null)
                if (chatMode) {
                    try {
                        val rj = JSONObject(result)
                        lastTurnEndedWithEos = rj.optBoolean("stopped_by_eos", false)
                        // ★要点だけ別に出す。結果 JSON は長すぎて logcat が切る。
                        Log.i(TAG, "chat turn: promptLen=" + rj.optInt("prompt_len", -1) +
                                " reused=" + rj.optInt("reused_tokens", -1) +
                                " filled=" + rj.optInt("prefill_tokens", -1) +
                                " eos=" + lastTurnEndedWithEos +
                                " new=" + (rj.optJSONArray("new_ids")?.length() ?: -1) +
                                " abs_pos=" + rj.optInt("abs_pos", -1) +
                                " pinned=" + rj.optInt("pinned_steps", -1))
                        val arr = rj.optJSONArray("new_ids")
                        val got = ArrayList<Int>(arr?.length() ?: 0)
                        if (arr != null) for (i in 0 until arr.length()) got.add(arr.getInt(i))
                        finishChatTurn(got)
                    } catch (t: Throwable) {
                        Log.w(TAG, "chat turn failed: ${t.message}")
                    }
                }
                showFinalStats(result)
                // テスト用の自動送信（インテント引数 prompts があるときだけ）
                if (autoPrompts.isNotEmpty()) ui.postDelayed({ sendNextAuto() }, 300)
                else if (intent?.hasExtra("prompts") == true) dumpAutoChat()
            }
        }
    }

    private val pollProgress = object : Runnable {
        override fun run() {
            // ★生成が終わっていたら何もしない。最後の1回が `showFinalStats` の後に走ると、
            //   統計が**1行の簡易表示で上書きされる**（2026-09-14 に画面で確認）。
            if (!generating) return
            try {
                val p = JSONObject(NativeBridge.nativeProgress())
                val rev = p.optLong("rev", -1L)
                if (rev != lastRev) {
                    lastRev = rev
                    val arr = p.optJSONArray("ids")
                    val ids = ArrayList<Int>(arr?.length() ?: 0)
                    if (arr != null) for (i in 0 until arr.length()) ids.add(arr.getInt(i))
                    renderOutput(ids)
                    if (p.has("median_ms")) {
                        val med = p.optDouble("median_ms")
                        statsView.text = String.format(
                            "%.1f ms/token · %.1f tok/s · %d tokens",
                            med, 1000.0 / med, ids.size
                        )
                    }
                }
            } catch (t: Throwable) {
                Log.w(TAG, "progress read failed: ${t.message}")
            }
            if (generating) ui.postDelayed(this, POLL_MS)
        }
    }

    /// 応答が確定したら会話ログへ入れる。
    private fun finishChatTurn(newIds: List<Int>) {
        val tok = tokenizer ?: return
        val raw = if (newIds.isEmpty()) "" else tok.decode(newIds)
        val ans = raw.replace("<|im_end|>", "").replace("<|endoftext|>", "").trim()
        chatLog.append("端末: ").append(ans).append("\n\n")
        chatTurns += 1
        Log.i(TAG, "chat answer turn=$chatTurns: $ans")
        promptText = chatLog.toString()
        outputView.text = promptText
        outputScroll.post { outputScroll.fullScroll(View.FOCUS_DOWN) }
    }

    private fun renderOutput(newIds: List<Int>) {
        val tok = tokenizer ?: return
        // ★対話モードで生成が終わったあとは触らない。
        //   進捗ポーリングの最後の1回が `finishChatTurn` の後に走ると、
        //   会話ログの下に生の生成文が二重に出る（2026-09-14 に画面で確認）。
        if (chatMode && !generating) return
        val generated = (if (newIds.isEmpty()) "" else tok.decodeStreaming(newIds))
            // 特殊トークンは**画面に出さない**。会話の区切りであって本文ではない。
            .replace("<|im_end|>", "").replace("<|endoftext|>", "")
        val sb = SpannableStringBuilder()
        sb.append(promptText)
        sb.setSpan(ForegroundColorSpan(Color.parseColor("#8A8A8E")), 0, promptText.length,
            Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        val start = sb.length
        sb.append(generated)
        sb.setSpan(StyleSpan(Typeface.BOLD), start, sb.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        outputView.text = sb
        outputScroll.post { outputScroll.fullScroll(View.FOCUS_DOWN) }
    }

    /**
     * 確定した統計を出す。
     *
     * **充填（prefill）の時間は生成トークンあたりの時間に混ぜない。**
     * 混ぜると「1トークンあたり」が実際より速く見える（あるいは遅く見える）。
     * 内訳として別に出す。
     */
    private fun showFinalStats(resultJson: String) {
        try {
            val r = JSONObject(resultJson)
            if (!r.optBoolean("ok", false)) {
                val err = r.optString("error", "unknown")
                statsView.text = if (err == "cancelled") "中断しました" else "失敗: $err"
                return
            }
            val step = r.optJSONObject("step_ms_stats")
            val ort = r.optJSONObject("ort_run_ms")
            val med = step?.optDouble("median") ?: 0.0
            val ortMed = ort?.optDouble("median") ?: 0.0
            val n = r.optJSONArray("new_ids")?.length() ?: 0
            val prefill = r.optDouble("prefill_ms", 0.0)
            val promptLen = r.optInt("prompt_len", 0)
            // 再利用できたぶんも出す。**対話では充填が伸びないことが見える。**
            val reused = r.optInt("reused_tokens", 0)
            val filled = r.optInt("prefill_tokens", promptLen)
            val extra = if (reused > 0) "・再利用 " + reused else ""
            // ★会話がキャッシュの長さを超えたら、古い発話から忘れていくことを画面で知らせる
            //   （1.5B は KV256。12往復ほどで超える。EdgeSLM_AndroidScale Phase 3）
            val absPos = r.optInt("abs_pos", 0)
            val cl = r.optInt("cache_len", 0)
            // ★PC から読むための1行。結果 JSON は describe を含んで長く、logcat の1行上限で速度の欄が切れる
            // 1ステップの内訳（準備 / 推論の呼び出し全体 / 後処理、中央値）と zero_copy の有無も出す（Phase 6）
            val bd = r.optJSONObject("breakdown_ms")
            val bdm = { k: String -> bd?.optJSONObject(k)?.optDouble("median") ?: -1.0 }
            Log.i(TAG, String.format(java.util.Locale.ROOT,
                "turn stats: median_ms=%.1f ort_ms=%.1f new=%d prefill_ms=%.0f filled=%d reused=%d abs_pos=%d" +
                    " zc=%s prep_ms=%.2f run_ms=%.2f post_ms=%.2f",
                med, ortMed, n, prefill, filled, reused, absPos,
                r.optBoolean("zero_copy", false), bdm("prep"), bdm("run_wall"), bdm("post")))
            val window = if (cl in 1 until absPos)
                System.lineSeparator() + "会話が窓（${cl} トークン）を超えました: 古い発話から忘れます" +
                    (if (r.optBoolean("pin_pos0", false)) "（先頭の区切りだけは保持）" else "")
            else ""
            statsView.text = String.format(
                "%.1f ms/token · %.1f tok/s · %d tokens" + System.lineSeparator() +
                    "うち OrtRun %.1f ms · 充填 %.0f ms（%d トークン%s）",
                med, if (med > 0) 1000.0 / med else 0.0, n, ortMed, prefill, filled, extra
            ) + window
        } catch (t: Throwable) {
            Log.w(TAG, "stats parse failed: ${t.message}")
        }
    }

    companion object {
        private const val TAG = "EdgeSLMKvDemo"
        private const val MODEL_REL = "models/ops/qwen_dec24_epctx.onnx"
        /// 対話モードの system prompt。**短く保つ**——KVキャッシュの容量を食うため。
        private const val SYSTEM_PROMPT = "あなたは車載アシスタントです。短く答えてください。"
        private const val COS_TABLE = "inputs/cos_table_l64.bin"
        private const val SIN_TABLE = "inputs/sin_table_l64.bin"
        private const val OVERRIDE_REL = "jobs/kv_demo.json"
        // これがあれば画面にモード切り替えが出る。無ければ従来どおり単一モード
        private const val MODES_REL = "jobs/kv_modes.json"
        /// assistant 開始直後に差し込む定型句を指す鍵（Qwen3 の空 think ブロック用）
        private const val ASSISTANT_PREFIX_KEY = "assistant_prefix"
        private const val POLL_MS = 50L
        private const val MAX_NEW = 32
        /// 対話モードの生成上限。**32 だと文が途中で切れる**（実機で確認）。
        private const val MAX_NEW_CHAT = 96
        private val STOP_IDS = listOf(151645, 151643)

        // 窓ずらし版と同じプリセット（比較しやすいように揃えてある）
        /// 対話モードのプリセット。**日本語で試せるようにしてある**
        /// （端末の IME を使わずに日本語を入れられる）。
        private val CHAT_PRESETS = listOf(
            "エアコン" to "エアコンを22度にして",
            "音楽" to "音楽をかけて",
            "確認" to "さっき何度にしたっけ？",
            "英語" to "What is the capital of France?"
        )

        private val PRESETS = listOf(
            "France" to "The capital city of Japan is Tokyo. The capital city of France is",
            "planet" to "The largest planet in our solar system is",
            "code" to "Remember this fact: the access code is 8317. Now here is some " +
                "unrelated text about the weather. Question: what is the access code? " +
                "The access code is",
            "日本語" to "むかしむかし、あるところに"
        )
    }
}
