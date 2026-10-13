package org.edgeslm.demo

import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * 端末側トークナイザが HuggingFace と同じ ID 列を出すかを**端末上で**確かめる。
 *
 * ## なぜ端末で測るのか
 *
 * ホスト側で「同じアルゴリズムを Python に移植して突き合わせる」ことはできるが、
 * それは**移植が正しい前提**での確認にすぎない。実際に動くのは Android 上の
 * Kotlin + ICU 正規表現 + `java.text.Normalizer` であって、そこで測らないと
 * 「ホストで通ることと端末で動くことは別」という本プロジェクトで何度も踏んだ型に
 * またはまる（C-7 / C-10 / I/O 型 / err 1002）。
 *
 * 特に ICU 正規表現は素の OpenJDK と別物である（`(?U)` が構文エラーになる等）。
 * **ここを端末で通しておくと、その差が効いていないことの根拠になる。**
 *
 * ## 使い方（adb から）
 *
 *     files/tokenizer/golden.jsonl  ... host/make_tokenizer_assets.py が生成
 *     files/jobs/tokenize.json      ... {"job_id":"tok","task":"tokenize"}
 *
 *     am start -n org.edgeslm.demo/.RunnerActivity -a org.edgeslm.demo.RUN \
 *              --es job jobs/tokenize.json
 *
 * 結果は `files/out/last_result.json` に出る。**モデルは開かない**
 * （トークナイズに NPU は要らないため）。
 */
object TokenizerCheck {

    const val TAG = "EdgeSLMRunner"

    /**
     * golden.jsonl の各行 `{"name":..., "text":..., "ids":[...]}` を照合する。
     *
     * 合否は**ID 列の完全一致**で判定する。部分一致や長さ一致では意味がない
     * （1トークンずれれば以降の生成が別物になる）。
     */
    fun run(baseDir: File, job: JSONObject): JSONObject {
        val result = JSONObject()
        result.put("job_id", job.optString("job_id", "tokenize"))
        result.put("task", "tokenize")

        val tokDir = File(baseDir, job.optString("tokenizer_dir", "tokenizer"))
        val golden = File(tokDir, job.optString("golden", "golden.jsonl"))

        val tok = try {
            Qwen2Tokenizer.load(tokDir)
        } catch (t: Throwable) {
            result.put("ok", false)
            result.put("error", "tokenizer load failed: ${t.message}")
            return result
        }

        // 任意の文字列をその場でトークナイズするだけの使い方も残す（デバッグ用）
        val texts = job.optJSONArray("texts")
        if (texts != null) {
            val arr = JSONArray()
            for (i in 0 until texts.length()) {
                val s = texts.getString(i)
                val o = JSONObject()
                o.put("text", s)
                o.put("ids", JSONArray(tok.encode(s).toList()))
                arr.put(o)
            }
            result.put("encoded", arr)
        }

        if (!golden.isFile) {
            result.put("ok", texts != null)
            if (texts == null) result.put("error", "golden not found: ${golden.absolutePath}")
            return result
        }

        var total = 0
        var pass = 0
        val failures = JSONArray()
        golden.forEachLine { line ->
            if (line.isBlank()) return@forEachLine
            val o = JSONObject(line)
            val text = o.getString("text")
            val expArr = o.getJSONArray("ids")
            val expected = IntArray(expArr.length()) { expArr.getInt(it) }
            val got = tok.encode(text)
            total += 1
            if (got.contentEquals(expected)) {
                pass += 1
            } else {
                // 失敗は**中身を出す**。件数だけでは直せない。
                val f = JSONObject()
                f.put("name", o.optString("name"))
                f.put("text", text)
                f.put("expected", JSONArray(expected.toList()))
                f.put("got", JSONArray(got.toList()))
                failures.put(f)
            }
        }

        // 往復（encode → decode）も見る。decode が壊れていれば表示が壊れる。
        var roundTrip = 0
        var roundTripOk = 0
        golden.forEachLine { line ->
            if (line.isBlank()) return@forEachLine
            val o = JSONObject(line)
            val text = o.getString("text")
            val expArr = o.getJSONArray("ids")
            val ids = (0 until expArr.length()).map { expArr.getInt(it) }
            roundTrip += 1
            // NFC 正規化を通すので、比較対象も正規化後の文字列にする
            val want = java.text.Normalizer.normalize(text, java.text.Normalizer.Form.NFC)
            if (tok.decode(ids) == want) roundTripOk += 1
        }

        result.put("ok", pass == total && roundTripOk == roundTrip)
        result.put("total", total)
        result.put("pass", pass)
        result.put("fail", total - pass)
        result.put("decode_total", roundTrip)
        result.put("decode_pass", roundTripOk)
        result.put("failures", failures)
        Log.i(TAG, "tokenize check: encode $pass/$total, decode $roundTripOk/$roundTrip")
        return result
    }
}
