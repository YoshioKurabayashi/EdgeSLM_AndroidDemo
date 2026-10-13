package org.edgeslm.demo

import java.io.File

/**
 * Qwen2 の byte-level BPE トークナイザ（端末側実装）。
 *
 * これがあることで「トークナイズも端末内」＝**ローカル完結**になる。
 * 従来はホスト（PC）でトークナイズしてトークンIDを端末へ送っていた。
 *
 * 依存を足していない（外部ライブラリなし）。読み込む資材は APK に埋めず
 * `files/tokenizer/` に置く——差し替えに APK の再ビルドが要らないようにするため。
 *
 *   vocab.txt    1行1トークン。**行番号がそのままID**
 *   specials.tsv `ID<TAB>内容`。<|im_end|> などの追加トークン（byte-level 変換を通さない）
 *   merges.txt   BPE のマージ規則を優先度順に1行1件
 *
 * トークナイズは CPU で行う。NPU に載せる意味はない（短い文字列の辞書引き）。
 *
 * 処理順は HuggingFace と同じ: **特殊トークンの切り出し → NFC 正規化 → pre-tokenize
 * → ByteLevel 写像 → BPE**。HF との ID 列一致は `golden.jsonl` で端末上で確認できる
 * （`task: "tokenize"` のジョブ。`TokenizerCheck.kt`）。
 */
class Qwen2Tokenizer private constructor(
    private val idToToken: Array<String>,
    private val tokenToId: HashMap<String, Int>,
    private val ranks: HashMap<String, Int>,
    private val specials: List<Pair<String, Int>>,   // 内容 → ID（長い順に並べてある）
    private val idToSpecial: HashMap<Int, String>,
) {

    private val bpeCache = HashMap<String, List<String>>()

    // ---------------------------------------------------------------- encode

    /** 文字列 → トークンID列。特殊トークンは文字列として書かれていれば特殊として扱う。 */
    fun encode(text: String): IntArray {
        val out = ArrayList<Int>()
        encodeInto(text, out)
        return out.toIntArray()
    }

    private fun encodeInto(text: String, out: MutableList<Int>) {
        if (text.isEmpty()) return

        // 特殊トークン（<|im_start|> 等）が書かれていたら、そこで切って特殊IDにする。
        // HuggingFace の既定挙動に合わせるため。
        var bestPos = -1
        var bestSpecial: Pair<String, Int>? = null
        for (sp in specials) {
            val p = text.indexOf(sp.first)
            if (p >= 0 && (bestPos < 0 || p < bestPos)) {
                bestPos = p
                bestSpecial = sp
            }
        }
        if (bestSpecial != null) {
            encodeOrdinary(text.substring(0, bestPos), out)
            out.add(bestSpecial.second)
            encodeInto(text.substring(bestPos + bestSpecial.first.length), out)
            return
        }
        encodeOrdinary(text, out)
    }

    private fun encodeOrdinary(rawText: String, out: MutableList<Int>) {
        if (rawText.isEmpty()) return

        // ★NFC 正規化。`tokenizer.json` の `normalizer: {"type": "NFC"}` に対応する。
        //
        // 入れる位置は**特殊トークンで切った後・pre-tokenize の前**である。
        // HuggingFace の処理順（追加語彙の切り出し → normalizer → pre_tokenizer → BPE）
        // と同じにするため。
        //
        // これが無いと、**分解済み（NFD）で入ってきた文字で HF と ID 列が変わる**。
        // 2026-09-13 に 31ケースで実測した結果:
        //   NFC なし 27/31 一致 / NFC あり 31/31 一致
        // 外れていたのは4件すべて分解済み入力だった:
        //   "が"（濁点が別文字）   HF [28195,…] / 旧実装 [31049,3402,247,…]
        //   "café"              HF [924,58858,…] / 旧実装 [924,1859,53839,…]
        //   "한"（ハングル分解） HF [23573] / 旧実装 [146073,146679,147245]
        //   "á̂"（結合記号）  HF [1953,…] / 旧実装 [64,53839,…]
        // 合成済み（通常の IME 出力）なら元から一致するので、**普段は症状が出ない**。
        // だからこそ検証しないと気付けない部分である。
        //
        // Unicode の分解表を自前で持つ必要はない。`java.text.Normalizer` は
        // Android に標準で入っている（追加依存ゼロ）。
        val text = java.text.Normalizer.normalize(rawText, java.text.Normalizer.Form.NFC)

        var cursor = 0
        val m = PATTERN.matcher(text)
        while (m.find()) {
            // 正規表現に拾われない隙間が出たら捨てずに処理する（黙って文字を落とさない）
            if (m.start() > cursor) emitPiece(text.substring(cursor, m.start()), out)
            emitPiece(m.group(), out)
            cursor = m.end()
        }
        if (cursor < text.length) emitPiece(text.substring(cursor), out)
    }

    private fun emitPiece(piece: String, out: MutableList<Int>) {
        if (piece.isEmpty()) return
        // ByteLevel: UTF-8 バイト列にしてから、1バイトを1文字へ写像する
        val sb = StringBuilder()
        for (b in piece.toByteArray(Charsets.UTF_8)) {
            sb.append(BYTE_TO_CHAR[b.toInt() and 0xFF])
        }
        for (sym in bpe(sb.toString())) {
            val id = tokenToId[sym]
            if (id != null) {
                out.add(id)
            } else {
                // 語彙に無い記号は1文字ずつ落とす（ここに来るのは辞書が壊れている場合だけ）
                for (c in sym) tokenToId[c.toString()]?.let { out.add(it) }
            }
        }
    }

    /**
     * BPE 本体。
     *
     * **選んだペアはその走査で全ての出現箇所をまとめて結合する**（GPT-2 の原実装と同じ）。
     * 左端の1箇所だけ結合する実装にすると、結合で生まれた新しいペアの方が
     * 優先度が高い場合に順序が変わり、HuggingFace と違うID列になる。
     */
    private fun bpe(token: String): List<String> {
        bpeCache[token]?.let { return it }
        var word = ArrayList<String>(token.length)
        for (c in token) word.add(c.toString())
        if (word.size < 2) {
            bpeCache[token] = word
            return word
        }

        while (true) {
            var bestRank = Int.MAX_VALUE
            var bestA = ""
            var bestB = ""
            for (i in 0 until word.size - 1) {
                val r = ranks[word[i] + " " + word[i + 1]] ?: continue
                if (r < bestRank) {
                    bestRank = r
                    bestA = word[i]
                    bestB = word[i + 1]
                }
            }
            if (bestRank == Int.MAX_VALUE) break

            val next = ArrayList<String>(word.size)
            var i = 0
            while (i < word.size) {
                if (i < word.size - 1 && word[i] == bestA && word[i + 1] == bestB) {
                    next.add(bestA + bestB)
                    i += 2
                } else {
                    next.add(word[i])
                    i += 1
                }
            }
            word = next
            if (word.size < 2) break
        }
        bpeCache[token] = word
        return word
    }

    // ---------------------------------------------------------------- decode

    /** トークンID列 → バイト列。UTF-8 の途中で切れることがあるのでバイトで返す。 */
    fun decodeToBytes(ids: List<Int>): ByteArray {
        val buf = java.io.ByteArrayOutputStream()
        for (id in ids) {
            val sp = idToSpecial[id]
            if (sp != null) {
                buf.write(sp.toByteArray(Charsets.UTF_8))
                continue
            }
            val t = idToToken.getOrNull(id) ?: continue
            for (c in t) {
                val b = CHAR_TO_BYTE[c]
                if (b != null) buf.write(b)
            }
        }
        return buf.toByteArray()
    }

    /**
     * 逐次表示用のデコード。
     *
     * 日本語は1文字3バイトなので、トークンの切れ目が**文字の途中**に来る。
     * そのまま String にすると壊れた文字（`�`）が出て、次のトークンで直る、という
     * ちらつきが起きる。**末尾の不完全なバイト列は出さずに持ち越す。**
     */
    fun decodeStreaming(ids: List<Int>): String {
        val bytes = decodeToBytes(ids)
        var end = bytes.size
        // 末尾から最大3バイト見て、途中で切れた UTF-8 シーケンスなら落とす
        var i = bytes.size - 1
        var back = 0
        while (i >= 0 && back < 4) {
            val b = bytes[i].toInt() and 0xFF
            if (b and 0x80 == 0) break                       // ASCII。切れていない
            if (b and 0xC0 == 0xC0) {                        // 先頭バイト
                val need = when {
                    b and 0xF8 == 0xF0 -> 4
                    b and 0xF0 == 0xE0 -> 3
                    else -> 2
                }
                if (bytes.size - i < need) end = i           // 足りない＝持ち越す
                break
            }
            i -= 1
            back += 1
        }
        return String(bytes, 0, end, Charsets.UTF_8)
    }

    fun decode(ids: List<Int>): String = String(decodeToBytes(ids), Charsets.UTF_8)

    // ---------------------------------------------------------------- 読み込み

    companion object {
        /**
         * pre-tokenizer の分割規則。`tokenizer.json` の Split パターンをそのまま移した。
         *
         * **Android の正規表現は ICU 実装**であり、素の Java（OpenJDK）とは別物である。
         * 効いてくる違いが2つある:
         *  - `(?U)`（UNICODE_CHARACTER_CLASS）は **ICU では構文エラーになる**。
         *    実際に `PatternSyntaxException: near index 3` で落ちた（2026-09-13）
         *  - その代わり **ICU の `\s` は元から Unicode 対応**（`[\t\n\f\r\p{Z}]`）で、
         *    全角スペースも空白として扱う。HuggingFace 側（Rust regex）と同じ挙動になる
         * つまり Android ではフラグを付けない方が正しい。
         */
        private val PATTERN: java.util.regex.Pattern = java.util.regex.Pattern.compile(
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)" +
                "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+" +
                "|\\p{N}" +
                "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*" +
                "|\\s*[\\r\\n]+" +
                "|\\s+(?!\\S)" +
                "|\\s+"
        )

        private val BYTE_TO_CHAR: CharArray = buildByteToChar()
        private val CHAR_TO_BYTE: HashMap<Char, Int> = HashMap<Char, Int>().also { m ->
            for (b in 0..255) m[BYTE_TO_CHAR[b]] = b
        }

        /**
         * GPT-2 由来の「バイト↔表示可能文字」写像。
         * 表示可能な ASCII 等はそのまま、それ以外は U+0100 以降へ退避する。
         * これにより語彙の文字列に制御文字・改行・空白が現れなくなる
         * （＝1行1トークンのテキストで語彙を持てる）。
         */
        private fun buildByteToChar(): CharArray {
            val out = CharArray(256)
            val direct = ArrayList<Int>()
            for (b in 0x21..0x7E) direct.add(b)
            for (b in 0xA1..0xAC) direct.add(b)
            for (b in 0xAE..0xFF) direct.add(b)
            var n = 0
            for (b in 0..255) {
                out[b] = if (direct.contains(b)) b.toChar() else (256 + n++).toChar()
            }
            return out
        }

        /** `files/tokenizer/` から読み込む。失敗したら例外を投げる（黙って壊れた状態で動かさない）。 */
        fun load(dir: File): Qwen2Tokenizer {
            val vocabFile = File(dir, "vocab.txt")
            val mergesFile = File(dir, "merges.txt")
            val specialsFile = File(dir, "specials.tsv")
            for (f in listOf(vocabFile, mergesFile, specialsFile)) {
                if (!f.isFile) throw IllegalStateException("tokenizer asset missing: ${f.absolutePath}")
            }

            val tokens = vocabFile.readLines()
            val idToToken = Array(tokens.size) { tokens[it] }
            val tokenToId = HashMap<String, Int>(tokens.size * 2)
            for (i in tokens.indices) tokenToId[tokens[i]] = i

            val ranks = HashMap<String, Int>(200_000)
            var rank = 0
            mergesFile.forEachLine { line ->
                if (line.isNotEmpty()) ranks[line] = rank++
            }

            val specials = ArrayList<Pair<String, Int>>()
            val idToSpecial = HashMap<Int, String>()
            specialsFile.forEachLine { line ->
                val t = line.indexOf('\t')
                if (t > 0) {
                    val id = line.substring(0, t).toInt()
                    val content = line.substring(t + 1)
                    specials.add(content to id)
                    idToSpecial[id] = content
                }
            }
            // 長いものから照合する（<|im_start|> と <|im_end|> のような包含関係に備える）
            specials.sortByDescending { it.first.length }

            return Qwen2Tokenizer(idToToken, tokenToId, ranks, specials, idToSpecial)
        }
    }
}
