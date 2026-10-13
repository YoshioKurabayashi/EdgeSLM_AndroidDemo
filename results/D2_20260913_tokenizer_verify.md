# D2: 端末側トークナイザを HF と突き合わせる（NFC 正規化の欠落を発見・修正）

| | |
|---|---|
| 実施日 | 2026-09-13 |
| 端末 | Sony Xperia 5 III SOG05（SM8350 / HTP v68） |
| 対象 | `demo_app/.../Qwen2Tokenizer.kt`（Kotlin 実装 / 外部ライブラリなし） |
| 目的 | 「端末でトークナイズできている」ことと「**HF と同じ ID 列を出す**」ことは別。後者を測る |

---

## 1. 結論

| | |
|---|---|
| 修正前 | **27/31 一致**（ホスト側で同一アルゴリズムを再現して測定） |
| 修正後（端末実測） | **encode 31/31・decode 往復 31/31 一致** |
| 真因 | `tokenizer.json` の `normalizer: {"type": "NFC"}` が端末側に無かった |
| 修正 | `java.text.Normalizer` を1行。**追加依存ゼロ**（Android 標準） |
| 既存機能への影響 | なし（`generate` は検証済みの列を再現） |

> **「端末で動いている」を「正しい」と読まないこと。**
> 合成済み文字（通常の IME 出力）では元から一致していたので、
> **測らなければ永久に気付かない**種類の欠落だった。

---

## 2. 何が欠けていたか

Qwen2.5 のトークナイズは4段で、**HF はこの順**で処理する。

```
追加語彙（<|im_end|> 等）の切り出し
  ↓
NFC 正規化          ← ここが端末側に無かった
  ↓
pre-tokenize（正規表現で単語に切る）
  ↓
ByteLevel 写像 → BPE
```

NFC 正規化は「**同じ文字の別表現を1つに寄せる**」処理である。

| | コードポイント | 見え方 |
|---|---|---|
| 合成済み（NFC） | `U+304C` | が |
| 分解済み（NFD） | `U+304B U+3099`（か + 濁点） | が |

**画面上は同じ「が」だが、別の文字列である。** HF は NFC に寄せてから
トークナイズするので、端末側が寄せないと ID 列が変わる。

### 実際にずれていた4件

```
"が ぎ …"      HF [28195, 220, 124902, …]  / 旧 [31049, 3402, 247, …]
"café …"            HF [924, 58858, …]          / 旧 [924, 1859, 53839, …]
"한"      HF [23573]（1トークン）      / 旧 [146073, 146679, 147245]
"á̂̃b"    HF [1953, 136, 224, …]      / 旧 [64, 53839, 136, 224, …]
```

ハングルが分かりやすい。HF は**1トークン**、旧実装は**3トークン**になっていた。

### いつ踏むか

- Android の IME 出力は通常 NFC なので**普段は出ない**
- macOS 由来のテキスト（ファイル名は NFD）、一部のコピー&ペースト、
  外部データの取り込みで混ざる
- 混ざると**エラーにならず、静かに別の入力として扱われる**

---

## 3. 修正

`Qwen2Tokenizer.encodeOrdinary()` の入口に1行:

```kotlin
val text = java.text.Normalizer.normalize(rawText, java.text.Normalizer.Form.NFC)
```

### 入れる位置を実測で決めた

`encodeOrdinary`（＝**特殊トークンで切った後**・pre-tokenize の前）に入れてある。
これは HF の処理順と同じである。

当初「文字列全体を最初に正規化する」案もあったが、HF は追加語彙を先に切り出すので
順序が変わる。**両方をホスト側で試して 31/31 になる方を選んだ。**

### Unicode の分解表は要らなかった

当初は「NFC のために Unicode の正規化テーブル（分解・合成・CCC・quick-check）を
生成して端末へ持たせる」つもりでいた。**不要だった**——`java.text.Normalizer` は
Android に標準で入っている。

> **自前で持つ前に、プラットフォームが持っていないかを確認する。**

---

## 4. 端末での検証（`task: "tokenize"`）

ホストで同一アルゴリズムを Python に移植して測ることはできるが、
それは**移植が正しい前提**での確認にすぎない。実際に動くのは Android の
**ICU 正規表現**と `java.text.Normalizer` である。

このプロジェクトは「ホストで通ることと端末で動くことは別」を
C-7 / C-10 / I/O 型 / err 1002 で何度も踏んでいるので、**端末で測る**。

### 仕組み

| 追加 | 内容 |
|---|---|
| `host/make_tokenizer_assets.py` | `golden.jsonl` を追加生成（HF の正解 ID 列 31件。**分解済み入力を意図的に混ぜてある**） |
| `TokenizerCheck.kt`（新規） | `golden.jsonl` と照合。任意文字列の encode もできる |
| `RunnerService.kt` | `task == "tokenize"` を**モデルを開かずに**処理（トークナイズに NPU は不要。772MB のロードを省く） |
| `jobs/tokenize.json` | ジョブ定義 |

```bash
adb shell am start -n org.edgeslm.demo/.RunnerActivity \
  -a org.edgeslm.demo.RUN --es job jobs/tokenize.json
```

### 結果

```json
{"job_id":"tok_golden","task":"tokenize","ok":true,
 "total":31,"pass":31,"fail":0,
 "decode_total":31,"decode_pass":31,"failures":[]}
```

**encode 31/31・decode 往復 31/31。** 照合しているケース:

| 分類 | 狙い |
|---|---|
| 英語・短縮形（大文字/小文字） | 正規表現の `(?i:'s\|'t\|…)` が効くか |
| 数字 | `\p{N}` が**1文字ずつ**切れる規則 |
| 連続空白・行末空白・改行・タブ | `\s+(?!\S)` の後戻りが絡む最難所 |
| 全角スペース | **ICU の `\s` が Unicode 対応**か（素の OpenJDK と違う） |
| 日本語・中国語・韓国語・絵文字（ZWJ 連結） | マルチバイトと結合 |
| 特殊トークン | `<\|im_start\|>` 等の切り出し |
| **分解済み（NFD）5件** | **NFC 正規化の有無がここで出る** |

### 回帰確認

`generate` が**検証済みの列を再現する**ことも確かめた（`RunnerService` に分岐を
入れたため）:

```
prompt  : The capital city of Japan is Tokyo. The capital city of France is
new_ids : [12095, 13, 576, 6722, 3283, 315]
期待    : [12095, 13, 576, 6722, 3283, 315]   ← D1 で fp32 と 6/6 一致を確認済み
一致    : ★完全一致
```

### 前半と後半が繋がっていることの確認

`demo_gen.json` の `prompt_ids` は**ホストで HF が出した列**である。
同じ文を**端末で encode** して一致するか見た:

```
prompt_ids  : [785, 6722, 3283, 315, 6323, 374, 26194, 13, 576, 6722, 3283, 315, 9625, 374]
端末 encode : [785, 6722, 3283, 315, 6323, 374, 26194, 13, 576, 6722, 3283, 315, 9625, 374]
一致        : True
```

日本語・NFD 混在でも一致:

| テキスト | 端末トークン数 | HF と一致 |
|---|---:|---|
| `The capital city of Japan is Tokyo. The capital city of France is` | 14 | ○ |
| `日本の首都は東京です。フランスの首都は` | 11 | ○ |
| `café と が（分解済み）` | 12 | ○ |

> **ホストでトークナイズしていた部分を端末に置き換えても、同じ入力になる。**
> これで「ローカル完結」が**数値で**裏付けられた。

---

## 5. ホスト側の検証ツール（再現用）

`EdgeSLM_Android/app/pipeline/verify_device_tokenizer.py` に、
**Kotlin と同じ手順を Python で再現して HF と突き合わせる**スクリプトを置いた。

```bash
cd ../EdgeSLM_Android
python -m app.pipeline.verify_device_tokenizer \
    ../EdgeSLM_AndroidDemo/host/out
```

`NFC_IN_ORDINARY` を切り替えると「NFC なし 27/31 / あり 31/31」を再現できる。
**端末を触らずにアルゴリズムの当たりを付けられる**ので、仕様変更時の一次確認に使う。

ただし**最終判定は端末の `task: "tokenize"` で行う**（ICU 正規表現の差は
ホストでは出ない）。

---

## 6. 残る限界

| | |
|---|---|
| golden が31件 | 壊れやすい所は押さえたが網羅ではない。長文・多言語混在を増やす余地あり |
| 正規表現は ICU 実装に依存 | HF（Rust regex / onig）と**別実装**である。31件で一致したが、保証ではない |
| chat template は未実装 | `<\|im_start\|>` 等を自分で書けば通るが、テンプレート適用は端末側に無い |
| vocab 読み込み | `vocab.txt` 151,643 行 + `merges.txt` 151,387 行のパース。**起動時1回**で実測上問題なし |
