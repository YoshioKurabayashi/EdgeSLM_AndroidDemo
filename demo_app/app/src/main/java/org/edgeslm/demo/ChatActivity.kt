package org.edgeslm.demo

import android.app.Activity
import android.content.Context
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.net.ConnectivityManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.util.Log
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.view.WindowManager
import android.view.inputmethod.EditorInfo
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
 * 同僚に見せるためのチャット画面（EdgeSLM_AndroidScale の成果のデモ。2026-09-19）。
 *
 * ## 何を見せるか
 *
 * - **インターネットを使わず、スマホの NPU だけで LLM が会話する**
 * - **1.5B（賢い）と 0.5B（速い）を切り替えて比べられる**。起動時は 1.5B の対話
 * - 3ターン目で「さっき何度にしたっけ？」に答えられる（会話を覚えている）
 *
 * ## 既存の画面との関係
 *
 * 技術者向けの画面（窓ずらし版 [DemoActivity]・KV 版 [KvDemoActivity]）は**そのまま残す**。
 * この画面は同じネイティブの経路（`generate_kv`）を使い、見せ方だけを変える。
 * モデルの設定は `files/jobs/chat_models.json`（無ければ下の既定値）。
 *
 * ## 同時には開かない
 *
 * 1プロセスで NPU に置けるのは約1,710MiB。1.5B（1,481MiB）を開いたまま別の画面でモデルを開くと超える。
 * この画面はモデルを1つだけ持ち、切り替えるときは先に閉じてから開く。
 */
class ChatActivity : Activity() {

    // ---- 画面の部品 ----
    private lateinit var statusView: TextView
    private lateinit var chatList: LinearLayout
    private lateinit var chatScroll: ScrollView
    private lateinit var inputEdit: EditText
    private lateinit var sendButton: Button
    private lateinit var stopButton: Button
    private lateinit var resetButton: Button
    private val modelButtons = ArrayList<LinearLayout>()
    private val suggestionButtons = ArrayList<Button>()

    // ---- 状態 ----
    @Volatile private var handle: Long = 0L
    @Volatile private var tokenizer: Qwen2Tokenizer? = null
    @Volatile private var generating = false
    @Volatile private var loading = false
    private var models: List<ChatModel> = emptyList()
    private var modelIndex = 0
    private var systemPrompt = DEFAULT_SYSTEM
    private var suggestionRows: List<List<String>> = DEFAULT_SUGGESTIONS
    private var chatStarted = false            // 1ターン目を送ったか（2ターン目以降は差分だけ送る）
    private var turns = 0
    private var windowNoticeShown = false
    private var readyStrict = false            // 表示用: 全ノードを NPU に置けたか（strict で開けたか）
    private var readyLoadMs = 0L               // 表示用: 読み込みにかかった時間
    private val transcript = StringBuilder()   // テスト用に書き出す会話
    private val ui = Handler(Looper.getMainLooper())

    // 生成中の吹き出し
    private var liveBubble: TextView? = null
    private var liveStats: TextView? = null
    private var lastRev = -1L
    private var sendAt = 0L
    private var firstTokenAt = 0L
    @Volatile private var genThread: Thread? = null

    // ---- テスト用のインテント引数（渡さなければ普通の画面）----
    //   am start -n org.edgeslm.demo/.ChatActivity --es model 0.5b --es prompts "エアコンを22度にして|音楽をかけて"
    private val autoPrompts = ArrayDeque<String>()

    /** 画面に出すモデル1つ分の設定 */
    private data class ChatModel(val key: String, val label: String, val sub: String, val desc: String,
                                 val job: JSONObject)

    // ================================================================= 起動

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // デモ中に画面が消えないようにする（この画面を表示している間だけ）
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        loadConfig()
        intent?.getStringExtra("model")?.let { k ->
            val i = models.indexOfFirst { it.key == k }
            if (i >= 0) modelIndex = i
        }
        intent?.getStringExtra("prompts")?.split("|")?.map { it.trim() }?.filter { it.isNotEmpty() }
            ?.let { autoPrompts.addAll(it) }
        setContentView(buildUi())
        refreshModelButtons()
        openModel(modelIndex, first = true)
    }

    override fun onDestroy() {
        super.onDestroy()
        ui.removeCallbacksAndMessages(null)
        val h = handle
        handle = 0L
        if (h != 0L) {
            // 生成中に閉じられたら、止めてから（生成のスレッドが抜けるのを待ってから）解放する
            NativeBridge.nativeCancel(h)
            val g = genThread
            thread(name = "edgeslm-chat-close") {
                runCatching { g?.join(5000) }
                NativeBridge.nativeDestroy(h)
            }
        }
    }

    /** `files/jobs/chat_models.json` を読む。無ければ既定値（端末に置いてある 1.5B と 0.5B）。 */
    private fun loadConfig() {
        try {
            val f = File(getExternalFilesDir(null), CONFIG_REL)
            if (f.isFile) {
                val o = JSONObject(f.readText())
                systemPrompt = o.optString("system_prompt", DEFAULT_SYSTEM)
                // 候補の質問（相手に合わせて作り直さずに差し替えられる）。[[行1…], [行2…]] か ["…", …]
                o.optJSONArray("suggestions")?.let { arr ->
                    val rows = ArrayList<List<String>>()
                    val flat = ArrayList<String>()
                    for (i in 0 until arr.length()) {
                        val v = arr.get(i)
                        if (v is JSONArray) rows.add((0 until v.length()).map { v.getString(it) })
                        else flat.add(v.toString())
                    }
                    if (flat.isNotEmpty()) rows.add(flat)
                    if (rows.isNotEmpty()) suggestionRows = rows
                }
                val arr = o.getJSONArray("models")
                models = (0 until arr.length()).map { i ->
                    val m = arr.getJSONObject(i)
                    ChatModel(m.getString("key"), m.getString("label"), m.optString("sub"),
                              m.optString("desc"), m.getJSONObject("job"))
                }
            }
        } catch (t: Throwable) {
            Log.w(TAG, "$CONFIG_REL ignored: ${t.javaClass.simpleName}: ${t.message}")
        }
        if (models.isEmpty()) models = defaultModels()
    }

    private fun defaultModels(): List<ChatModel> = listOf(
        ChatModel("1.5b", "1.5B", "賢い", "Qwen2.5-1.5B-Instruct · 28層 · 1,481MB",
            JSONObject("""{"model":"models/ops/q15kc28_epctx.onnx",
                "const_inputs":{"cos_table":"inputs/cos_table_q15_l2048.bin","sin_table":"inputs/sin_table_q15_l2048.bin"},
                "embed_table":"models/ops/q15_embed_bf16.bin","embed_dtype":"bf16",
                "pos0_kv":"inputs/q15kc28_pos0_kv.bin","pin_pos0":true,"zero_copy":true}""")),
        ChatModel("0.5b", "0.5B", "速い", "Qwen2.5-0.5B-Instruct · 24層 · 772MB",
            JSONObject("""{"model":"models/ops/qwen_dec24c512_epctx.onnx",
                "const_inputs":{"cos_table":"inputs/cos_table_l2048.bin","sin_table":"inputs/sin_table_l2048.bin"},
                "pin_pos0":true,"zero_copy":true}"""))
    )

    // ================================================================= 画面

    private fun dp(v: Float): Int =
        TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, resources.displayMetrics).toInt()

    private fun rounded(color: Int, radiusDp: Float, strokeColor: Int? = null): GradientDrawable =
        GradientDrawable().apply {
            setColor(color)
            cornerRadius = dp(radiusDp).toFloat()
            if (strokeColor != null) setStroke(dp(2f), strokeColor)
        }

    private fun buildUi(): View {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(BG)
            setPadding(dp(16f), dp(14f), dp(16f), dp(10f))
        }

        // 見出しと「新しい会話」
        val head = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
        val titles = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        titles.addView(TextView(this).apply {
            text = "オンデバイス AI チャット"
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 21f)
            setTypeface(typeface, Typeface.BOLD)
            setTextColor(INK)
        })
        titles.addView(TextView(this).apply {
            text = "インターネットを使わず、スマホの NPU だけで動いています"
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 12.5f)
            setTextColor(SUB)
        })
        head.addView(titles, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
        resetButton = Button(this).apply {
            text = "新しい会話"
            isAllCaps = false
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
            setOnClickListener { newConversation(notice = true) }
        }
        head.addView(resetButton, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT))
        root.addView(head)

        // モデルの切り替え（2つのカード）
        val modelRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(0, dp(10f), 0, dp(4f))
        }
        models.forEachIndexed { i, m ->
            val card = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER_HORIZONTAL
                setPadding(dp(8f), dp(8f), dp(8f), dp(8f))
                isClickable = true
                setOnClickListener { switchModel(i) }
            }
            card.addView(TextView(this).apply {
                text = "${m.label}（${m.sub}）"
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 17f)
                setTypeface(typeface, Typeface.BOLD)
                gravity = Gravity.CENTER
            })
            card.addView(TextView(this).apply {
                text = m.desc
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 10.5f)
                gravity = Gravity.CENTER
            })
            modelButtons.add(card)
            modelRow.addView(card, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f).apply {
                marginStart = if (i == 0) 0 else dp(6f)
                marginEnd = if (i == models.size - 1) 0 else dp(6f)
            })
        }
        root.addView(modelRow)

        // 状態（読み込み中 / NPU で動作中・通信の有無）
        statusView = TextView(this).apply {
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 12.5f)
            setTextColor(SUB)
            setPadding(dp(2f), dp(4f), 0, dp(6f))
        }
        root.addView(statusView)

        // 会話
        chatList = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(4f), dp(8f), dp(4f), dp(8f))
        }
        chatScroll = ScrollView(this).apply {
            background = rounded(Color.parseColor("#EEF0F4"), 14f)
            addView(chatList)
        }
        root.addView(chatScroll, LinearLayout.LayoutParams(MATCH_PARENT, 0, 1f))

        // 質問の候補（押すとそのまま送る）。行ごとに横へスクロール
        suggestionRows.forEachIndexed { r, row ->
            val sugRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
            row.forEach { s ->
                sugRow.addView(Button(this).apply {
                    text = s
                    isAllCaps = false
                    setTextSize(TypedValue.COMPLEX_UNIT_SP, 12.5f)
                    setOnClickListener { send(s) }
                    suggestionButtons.add(this)
                })
            }
            root.addView(HorizontalScrollView(this).apply {
                isHorizontalScrollBarEnabled = false
                setPadding(0, if (r == 0) dp(6f) else 0, 0, 0)
                addView(sugRow)
            })
        }

        // 入力
        val inputRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        inputEdit = EditText(this).apply {
            hint = "メッセージを入力"
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 16f)
            setSingleLine(true)
            imeOptions = EditorInfo.IME_ACTION_SEND
            setOnEditorActionListener { _, actionId, _ ->
                if (actionId == EditorInfo.IME_ACTION_SEND) { send(text.toString()); true } else false
            }
        }
        inputRow.addView(inputEdit, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
        sendButton = Button(this).apply {
            text = "送信"
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 15f)
            setOnClickListener { send(inputEdit.text.toString()) }
        }
        inputRow.addView(sendButton, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT))
        stopButton = Button(this).apply {
            text = "停止"
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 15f)
            visibility = View.GONE
            setOnClickListener { if (handle != 0L) NativeBridge.nativeCancel(handle) }
        }
        inputRow.addView(stopButton, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT))
        root.addView(inputRow)
        return root
    }

    private fun refreshModelButtons() {
        modelButtons.forEachIndexed { i, card ->
            val sel = i == modelIndex
            card.background = if (sel) rounded(ACCENT, 12f) else rounded(Color.WHITE, 12f, Color.parseColor("#C9CED8"))
            for (k in 0 until card.childCount) {
                (card.getChildAt(k) as? TextView)?.setTextColor(if (sel) Color.WHITE else INK)
            }
        }
    }

    private fun setBusy(busy: Boolean) {
        val ready = !busy && handle != 0L && tokenizer != null
        sendButton.isEnabled = ready
        suggestionButtons.forEach { it.isEnabled = ready }
        resetButton.isEnabled = !generating && !loading
        modelButtons.forEach { it.isEnabled = !generating && !loading; it.alpha = if (generating || loading) 0.6f else 1f }
        sendButton.visibility = if (generating) View.GONE else View.VISIBLE
        stopButton.visibility = if (generating) View.VISIBLE else View.GONE
        // ★停止は**最初の文字が出てから**押せる。プロンプトを読み込んでいる途中で止めると、
        //   キャッシュに発話の途中までしか入らず、次のターンの文脈が崩れる
        stopButton.isEnabled = generating && firstTokenAt > 0L
    }

    /**
     * 通信の有無（「インターネットを使っていない」ことを見せるため）。
     * このアプリはそもそも INTERNET 権限を持たない（マニフェストに無い。端末の dumpsys でも確認）。
     */
    private fun networkLabel(): String {
        val airplane = runCatching {
            Settings.Global.getInt(contentResolver, Settings.Global.AIRPLANE_MODE_ON, 0) == 1
        }.getOrDefault(false)
        val cm = getSystemService(Context.CONNECTIVITY_SERVICE) as? ConnectivityManager
        val net = runCatching { cm?.activeNetwork }.getOrNull()
        return when {
            airplane && net == null -> "機内モード（通信なし）"
            net == null -> "通信なし"
            else -> "通信は使っていません"
        }
    }

    /** 読み込みが終わった状態の表示（機内モードを切り替えて戻ってきたときにも書き直す） */
    private fun readyStatus(): String {
        val m = models[modelIndex]
        return "● ${m.label} を NPU で実行中" + (if (readyStrict) "（全ノード）" else "") +
            " · 読み込み ${"%.1f".format(readyLoadMs / 1000.0)} 秒 · ${networkLabel()}"
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        // クイック設定で機内モードを切り替えて閉じると、ここに戻ってくる
        if (hasFocus && handle != 0L && !loading) statusView.text = readyStatus()
    }

    // ================================================================= 会話の吹き出し

    private fun addUserBubble(text: String) {
        val tv = TextView(this).apply {
            this.text = text
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 16f)
            setTextColor(Color.WHITE)
            background = rounded(ACCENT, 16f)
            setPadding(dp(12f), dp(8f), dp(12f), dp(8f))
        }
        chatList.addView(tv, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT).apply {
            gravity = Gravity.END
            topMargin = dp(6f)
            marginStart = dp(48f)
        })
        scrollDown()
    }

    private fun addAssistantBubble(): Pair<TextView, TextView> {
        val box = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        val tv = TextView(this).apply {
            text = "…"
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 16f)
            setTextColor(INK)
            background = rounded(Color.WHITE, 16f)
            setPadding(dp(12f), dp(8f), dp(12f), dp(8f))
        }
        val st = TextView(this).apply {
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 11f)
            setTextColor(SUB)
            setPadding(dp(6f), dp(2f), 0, 0)
        }
        box.addView(tv, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT))
        box.addView(st, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT))
        chatList.addView(box, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT).apply {
            gravity = Gravity.START
            topMargin = dp(6f)
            marginEnd = dp(32f)
        })
        scrollDown()
        return tv to st
    }

    private fun addNotice(text: String) {
        chatList.addView(TextView(this).apply {
            this.text = text
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 11.5f)
            setTextColor(SUB)
            gravity = Gravity.CENTER
            setPadding(dp(8f), dp(8f), dp(8f), dp(4f))
        }, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        scrollDown()
    }

    private fun scrollDown() = chatScroll.post { chatScroll.fullScroll(View.FOCUS_DOWN) }

    // ================================================================= モデルを開く・切り替える

    private fun jobBase(m: ChatModel): JSONObject {
        val job = JSONObject().apply {
            put("job_id", "chat_ui")
            put("task", "generate_kv")
            put("mask_neg", -60.0)
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
        for (k in m.job.keys()) job.put(k, m.job.get(k))
        return job
    }

    private fun withPaths(job: JSONObject): String = JsonInject.addRuntimePaths(
        job.toString(),
        baseDir = getExternalFilesDir(null)!!.absolutePath,
        nativeLibDir = applicationInfo.nativeLibraryDir,
        overridesJson = null
    )

    private fun openModel(index: Int, first: Boolean) {
        val m = models[index]
        loading = true
        setBusy(true)
        statusView.text = "${m.label} のモデルを読み込み中…（数秒。この画面のままお待ちください）"
        statusView.setTextColor(SUB)
        thread(name = "edgeslm-chat-open") {
            try {
                if (tokenizer == null) {
                    tokenizer = Qwen2Tokenizer.load(File(getExternalFilesDir(null), "tokenizer"))
                }
                val old = handle
                if (old != 0L) {
                    handle = 0L
                    NativeBridge.nativeDestroy(old)
                }
                val t0 = System.currentTimeMillis()
                val h = NativeBridge.nativeCreate(withPaths(jobBase(m)))
                val loadMs = System.currentTimeMillis() - t0
                val d = JSONObject(NativeBridge.nativeDescribe(h))
                if (h == 0L) {
                    Log.i(TAG, "chat open failed: model=${m.key} error=${d.optString("error")}")
                    ui.post {
                        loading = false
                        statusView.text = "モデルを開けませんでした。ほかの画面でモデルを開いたままのときは、" +
                            "アプリを履歴から消してから開き直してください。\n${d.optString("error")}"
                        statusView.setTextColor(Color.parseColor("#B42318"))
                        setBusy(true)
                    }
                    return@thread
                }
                handle = h
                val npu = d.optBoolean("strict_no_cpu_fallback", false)
                Log.i(TAG, "chat ready: model=${m.key} load_ms=$loadMs strict=$npu net=${networkLabel()}")
                ui.post {
                    loading = false
                    readyStrict = npu
                    readyLoadMs = loadMs
                    statusView.text = readyStatus()
                    statusView.setTextColor(Color.parseColor("#1E7B45"))
                    setBusy(false)
                    if (first && chatList.childCount == 0) {
                        addNotice("下の候補を押すか、メッセージを入力してください")
                    }
                    if (autoPrompts.isNotEmpty()) ui.postDelayed({ sendNextAuto() }, 300)
                }
            } catch (t: Throwable) {
                Log.e(TAG, "open failed", t)
                ui.post {
                    loading = false
                    statusView.text = "初期化に失敗: ${t.javaClass.simpleName}: ${t.message}"
                    setBusy(true)
                }
            }
        }
    }

    private fun switchModel(index: Int) {
        if (generating || loading || index == modelIndex) return
        modelIndex = index
        refreshModelButtons()
        newConversation(notice = false)
        addNotice("${models[index].label}（${models[index].sub}）に切り替えました")
        openModel(index, first = false)
    }

    private fun newConversation(notice: Boolean) {
        if (generating) return
        chatStarted = false
        turns = 0
        windowNoticeShown = false
        transcript.setLength(0)
        chatList.removeAllViews()
        if (notice) addNotice("新しい会話を始めました")
    }

    // ================================================================= 送る・生成する

    private fun sendNextAuto() {
        if (generating || loading || handle == 0L) return
        val next = autoPrompts.removeFirstOrNull() ?: return
        send(next)
    }

    /**
     * 1ターン目は System Prompt を含む全文、2ターン目以降は**差分だけ**を送る
     * （前のターンの KV キャッシュを持ち回す。KvDemoActivity と同じ規則）。
     */
    private fun delta(user: String): String =
        if (!chatStarted)
            "<|im_start|>system\n$systemPrompt<|im_end|>\n<|im_start|>user\n$user<|im_end|>\n<|im_start|>assistant\n"
        else
            "<|im_end|>\n<|im_start|>user\n$user<|im_end|>\n<|im_start|>assistant\n"

    private fun send(raw: String) {
        val text = raw.trim()
        val tok = tokenizer ?: return
        if (text.isEmpty() || generating || loading || handle == 0L) return
        val ids = tok.encode(delta(text))
        if (ids.isEmpty()) return
        inputEdit.setText("")
        addUserBubble(text)
        transcript.append("あなた: ").append(text).append("\n")
        val (bubble, stats) = addAssistantBubble()
        liveBubble = bubble
        liveStats = stats
        stats.text = "考えています…"
        val job = jobBase(models[modelIndex]).apply {
            put("max_new", MAX_NEW)
            put("prompt_ids", JSONArray().also { a -> ids.forEach { a.put(it) } })
            put("stop_ids", JSONArray().also { a -> STOP_IDS.forEach { a.put(it) } })
            put("reuse_cache", true)
            put("reset_cache", !chatStarted)
            put("prompt_is_delta", chatStarted)
        }
        chatStarted = true
        generating = true
        setBusy(true)
        // ★いまの途中経過の版を覚えておき、それが変わるまで読まない。ジョブが始まる（begin で版が進み中身が空になる）前に
        //   読むと、前のターンのトークンを拾ってしまう（最初の版で「最初の文字まで 0.0 秒」と出た）
        lastRev = runCatching { JSONObject(NativeBridge.nativeProgress()).optLong("rev", -1L) }.getOrDefault(-1L)
        sendAt = System.currentTimeMillis()
        firstTokenAt = 0L
        ui.post(poll)
        val jobJson = withPaths(job)
        val h = handle
        genThread = thread(name = "edgeslm-chat-generate") {
            val result = NativeBridge.nativeRunJob(h, jobJson)
            ui.post { finishTurn(result) }
        }
    }

    /** 途中経過（50ms ごと）。選ばれたトークンを文字にして吹き出しへ */
    private val poll = object : Runnable {
        override fun run() {
            if (!generating) return
            try {
                val p = JSONObject(NativeBridge.nativeProgress())
                val rev = p.optLong("rev", -1L)
                if (rev != lastRev) {
                    lastRev = rev
                    val arr = p.optJSONArray("ids")
                    val ids = ArrayList<Int>(arr?.length() ?: 0)
                    if (arr != null) for (i in 0 until arr.length()) ids.add(arr.getInt(i))
                    if (ids.isNotEmpty()) {
                        if (firstTokenAt == 0L) {
                            firstTokenAt = System.currentTimeMillis()
                            stopButton.isEnabled = true
                        }
                        liveBubble?.text = clean(tokenizer?.decodeStreaming(ids) ?: "")
                        liveStats?.text = "生成中… ${ids.size} トークン"
                        scrollDown()
                    }
                }
            } catch (t: Throwable) {
                Log.w(TAG, "progress read failed: ${t.message}")
            }
            if (generating) ui.postDelayed(this, POLL_MS)
        }
    }

    private fun clean(s: String) = s.replace("<|im_end|>", "").replace("<|endoftext|>", "").trim()

    private fun finishTurn(result: String) {
        generating = false
        val endAt = System.currentTimeMillis()
        val bubble = liveBubble
        val stats = liveStats
        try {
            val r = JSONObject(result)
            if (!r.optBoolean("ok", false) && r.optString("error") != "cancelled") {
                bubble?.text = "（エラー: ${r.optString("error", "unknown")}）"
                stats?.text = ""
                setBusy(false)
                return
            }
            val arr = r.optJSONArray("new_ids")
            val ids = ArrayList<Int>(arr?.length() ?: 0)
            if (arr != null) for (i in 0 until arr.length()) ids.add(arr.getInt(i))
            val cancelled = r.optString("error") == "cancelled"
            val ans = clean(tokenizer?.decode(ids) ?: "")
            bubble?.text = ans.ifEmpty { if (cancelled) "（停止しました）" else "（応答なし）" }
            turns += 1
            transcript.append("端末: ").append(ans).append("\n\n")
            val med = r.optJSONObject("step_ms_stats")?.optDouble("median") ?: 0.0
            val ort = r.optJSONObject("ort_run_ms")?.optDouble("median") ?: 0.0
            val firstS = if (firstTokenAt > 0) (firstTokenAt - sendAt) / 1000.0 else Double.NaN
            val totalS = (endAt - sendAt) / 1000.0
            val tps = if (med > 0) 1000.0 / med else 0.0
            stats?.text = (if (cancelled) "停止しました · " else "") + String.format(java.util.Locale.ROOT,
                "%.1f トークン/秒 · 最初の文字まで %.1f 秒 · 全体 %.1f 秒 · 1トークン %.0fms（NPU %.0fms）",
                tps, firstS, totalS, med, ort)
            // ★改行は ⏎ にして1行で出す（logcat は改行で行が分かれ、PC 側が1行目しか拾えなかった。2026-09-19）
            Log.i(TAG, "chat answer turn=$turns: ${ans.replace("\r", "").replace("\n", "⏎")}")
            Log.i(TAG, String.format(java.util.Locale.ROOT,
                "chat stats turn=%d: model=%s first_s=%.2f total_s=%.2f median_ms=%.1f ort_ms=%.1f new=%d abs_pos=%d " +
                    "pinned=%d pin_src=%s carry=%d cancelled=%b",
                turns, models[modelIndex].key, firstS, totalS, med, ort, ids.size, r.optInt("abs_pos", -1),
                r.optInt("pinned_steps", 0), r.optString("pin_pos0_src", "-"), r.optInt("carry_tokens", -1), cancelled))
            // 会話がキャッシュの長さを超えたら一度だけ知らせる（古い発話から忘れていく）
            val absPos = r.optInt("abs_pos", 0)
            val cl = r.optInt("cache_len", 0)
            if (!windowNoticeShown && cl in 1 until absPos) {
                windowNoticeShown = true
                addNotice("会話が長くなったので、古い発話から順に忘れていきます（「新しい会話」でリセット）")
            }
        } catch (t: Throwable) {
            Log.w(TAG, "result parse failed: ${t.message}")
        }
        setBusy(false)
        scrollDown()
        if (autoPrompts.isNotEmpty()) ui.postDelayed({ sendNextAuto() }, 400)
        else if (intent?.hasExtra("prompts") == true) {
            runCatching { File(getExternalFilesDir(null), "out/chat_ui_chat.txt").writeText(transcript.toString()) }
            Log.i(TAG, "auto_chat done turns=$turns")
        }
    }

    companion object {
        private const val TAG = "EdgeSLMChat"
        private const val CONFIG_REL = "jobs/chat_models.json"
        private const val DEFAULT_SYSTEM = "あなたは車載アシスタントです。短く答えてください。"
        private const val MAX_NEW = 96
        private const val POLL_MS = 50L
        private val STOP_IDS = listOf(151645, 151643)
        // 1.5B は正しく、0.5B は誤りやすい質問を入れてある（2026-09-19 に実機で確かめた。EdgeSLM_AndroidScale P8）
        private val DEFAULT_SUGGESTIONS = listOf(
            listOf("エアコンを22度にして", "音楽をかけて", "さっき何度にしたっけ？"),
            listOf("富士山の高さは？", "12×8は？", "「おはよう」を英語にして", "What is the capital of France?")
        )
        private val BG = Color.parseColor("#F7F8FA")
        private val INK = Color.parseColor("#1F2430")
        private val SUB = Color.parseColor("#667085")
        private val ACCENT = Color.parseColor("#2563EB")
    }
}
