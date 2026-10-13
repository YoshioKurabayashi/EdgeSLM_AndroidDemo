# EdgeSLM_AndroidDemo

**Qwen2.5-0.5B を Xperia 5 III（Snapdragon 888 / HTP v68）の NPU で動かし、
端末単体で文章を生成して見せるためのデモアプリ。**

[EdgeSLM_Android](../EdgeSLM_Android)（検証プロジェクト）で「端末の NPU で生成できる」ところまで
到達した成果を、**同僚に一目で伝わる形**にしたもの。検証プロジェクトのファイルは1つも変更していない。

**同僚に見せるなら「AIチャット」の画面を使う**（2026-09-19 追加。Qwen2.5-**1.5B**（賢い）と **0.5B**（速い）を切り替えて会話する。
起動時は 1.5B）。台本・準備・見せる数字・想定質問・トラブル時の対処は **[DoC/デモ手順書.md](DoC/デモ手順書.md)**。
検証の記録は [EdgeSLM_AndroidScale P8](../EdgeSLM_AndroidScale/results/P8_20260919_chat_ui.md)。

## これは何が違うのか

| | 検証プロジェクト（既存） | **本デモ** |
|---|---|---|
| 生成ループ | PC が adb 越しに1トークンずつ叩く | **端末内（C++）で回す** |
| トークナイザ | PC 側（HuggingFace） | **端末内（Kotlin 実装）** |
| 1トークン | 約 4,800 ms | **32〜43 ms** |
| 操作 | コマンドライン | **画面に打ち込んで Generate** |
| PC | 必須 | **不要**（機内モードで動く） |

生成結果そのものは変わっていない。同じプロンプトで**検証済み経路とトークン列が完全一致**する
ことを確認してある（[`results/D1_20260913_generate_on_device.md`](results/D1_20260913_generate_on_device.md)）。

## 実測（Xperia 5 III SOG05 / 全ノード NPU）

```
起動          モデル 772MB のロード 0.9〜3.6 秒（初回のみ）+ 辞書 0.2〜0.9 秒
生成          32〜43 ms/token（実時間）、24〜31 tok/s
              うち OrtRun 30〜36 ms
配置          QNN HTP - EPContext - all nodes / strict_no_cpu_fallback = true
```

生成例（いずれも端末単体）:

```
The capital city of Japan is Tokyo. The capital city of France is
  → Paris. The capital city of the United States is Washington, D. C. ...

Python is a programming language. Java is a programming
  → language. It is a popular choice for beginners due to its simplicity ...

日本の首都は東京です。フランスの首都は
  → 、東京の南側に位置しています。その名の由来は、この地域の山脈が、
```

日本語も文字化けせず逐次表示される。

## 画面とモード

### 3つの画面（ランチャーに3つ並ぶ）

| アイコン | 画面 | 方式 | 速度 |
|---|---|---|---:|
| **AIチャット** | `ChatActivity` | KVキャッシュ decode（KV 画面と同じ経路）。**1.5B / 0.5B を切り替えるチャット**（同僚向け。2026-09-19） | 1.5B 66〜70 ms/token・0.5B 33〜37 ms/token |
| EdgeSLM Demo | `DemoActivity` | **窓ずらし**（毎回モデル全体を計算し直す） | 33〜42 ms/token |
| EdgeSLM KV | `KvDemoActivity` | **KVキャッシュ decode**（1トークンずつ） | 28.5 ms/token |

EdgeSLM Demo と EdgeSLM KV は画面下のボタンで相互に行き来できる。切り替えるときは**前の画面のセッションを先に解放してから**開く（切替 約1.4秒）。
3つとも同じアプリ（同じプロセス）なので、**ほかの画面がモデルを開いたまま AIチャット を開くと、NPU の上限を超えて開けない**
（そのときは最近使ったアプリの一覧からアプリを消して開き直す）。

### AIチャット（`ChatActivity`、2026-09-19）

- 設定は `files/jobs/chat_models.json`（リポジトリは `jobs/chat_models.json`。無ければコードの既定値）。**候補の質問（`suggestions`）・
  System Prompt・カードの表示・モデルの job を、アプリを作り直さずに変えられる**
- モデルは1つだけ持つ。カードを押すと、今のセッションを閉じてから次を開く（1.5B 2.2〜2.6秒・0.5B 約1.1秒）。会話は新しくなる
- 「停止」は**答えの文字が出始めてから**押せる（読み込み途中で止めると、キャッシュに発話の途中までしか入らず次の会話が崩れる）。止めても会話は続く
- ダークモードの自動切り替えなどの設定変更で画面を作り直さない（`configChanges`）。戻るボタンでは画面は閉じず裏へ回る（Android 12 以降の動き）
- 状態の行に「通信は使っていません」／「機内モード（通信なし）」を出す。**アプリは INTERNET 権限を持たない**（インストール済みの権限で確認）
- テスト用のインテント引数（渡さなければ普通の画面）:

```bash
adb shell am start -n org.edgeslm.demo/.ChatActivity --es model 0.5b --es prompts "エアコンを22度にして|さっき何度にしたっけ？"
```

| 引数 | 意味 |
|---|---|
| `model` | 起動時のモデル（`1.5b` / `0.5b`） |
| `prompts` | 各ターンが終わるたびに次の文を自動で送る（`\|` 区切り）。終わると会話を `out/chat_ui_chat.txt` に書き、logcat に `auto_chat done` を出す |

logcat（タグ `EdgeSLMChat`）には `chat ready`（読み込み時間・strict・通信）／`chat answer`／`chat stats`（最初の文字まで・全体・1トークン・位置0 を残した回数・持ち越したトークン数・停止したか）が出る。
PC からの駆動は `EdgeSLM_AndroidScale/app/pipeline/chat_ui_test.py`（`auto` / `ops`＝ボタンを押す14項目 / `relaunch`）。

理由（2026-09-19 に測り直した。[EdgeSLM_AndroidScale P0A](../EdgeSLM_AndroidScale/results/P0A_20260919_memory_ceiling.md)）:

- 以前は「772MB のセッションは同時に2つ開けない」と書いていたが、**772MB×2 は保持できる**（dma-buf 1,509MiB）
- 実際の壁は2つある。①**1プロセスで NPU に持てるのは約1,710MiB**まで（超えると error 1002）。Qwen2.5 KV512 と Qwen3 を同時に持つと超える。②**ロード中は一時的に binary の約2倍**を要求し、端末の MemFree が少ないと **lmkd に kill される**（2026-09-13 の SIGKILL はおそらくこれ）
- ②は**画面が前面にあれば起きなかった**（1.5B・1,481MiB で 25/25。アプリ9本を開いた負荷ありでも 10/10）。lmkd は前面のアプリを最後にしか選ばない。
  kill されたのは、ロード中に別の画面が前面へ出て、このアプリが裏（oom_score_adj 700）に回ったときだけだった（[P3](../EdgeSLM_AndroidScale/results/P3_20260919_demo.md) 2章）。
  **ロード中（数秒）は画面を切り替えない**

### 画面の中のモード切り替え

`files/jobs/demo_modes.json` を置くと、画面上部にモード行が出る。
**モデルは別ファイルなので、切り替えは JSON だけ。再ビルドは要らない。**

| ボタン | モデル | 位置づけ |
|---|---|---|
| 窓64 マスク **PER-CHANNEL** | `qwen_pm64pc_epctx.onnx` | 現行（既定） |
| 窓64 マスク PER-TENSOR | `qwen_pm64_epctx.onnx` | 重みが per-tensor だった頃の版 |
| 窓16 固定（最初の版） | `qwen24ext_epctx.onnx` | 一番最初に動いた版。**学習用に残してある** |

同じプロンプトで押し比べると差が見える。

#### KV 画面のモード（`files/jobs/kv_modes.json`）

| ボタン | モデル | 実測（対話モード・実機） |
|---|---|---|
| QWEN2.5 KV512 | `qwen_dec24c512_epctx.onnx`（24層 772MB） | **26.6 tok/s**（OrtRun 32.5ms / 充填 1409ms・37トークン）。`pin_pos0`（2026-09-19 追加。位置0 は端末で計算したもの）で **24ターン・822トークンまで崩れない**（無いと 512 を超えた次のターンで崩れる） |
| QWEN3 KV256 | `qwen3_dec28c256_epctx.onnx`（28層 872MB） | **10.4 tok/s**（OrtRun 77.9ms / 充填 3928ms・41トークン）。切替 1.7秒 |
| **QWEN2.5-1.5B KV256** | `q15kc28_epctx.onnx`（28層 1,481MiB。[EdgeSLM_AndroidScale](../EdgeSLM_AndroidScale)） | **約14 tok/s**（`zero_copy` で1ステップ 約70ms・OrtRun 約69ms / 最初の文字まで 1ターン目 約2.5秒・2ターン目以降 約1.4秒。`zero_copy` 無しは 約74〜78ms）。起動 2.4〜3.0秒、切替 2.6〜3.8秒。**3ターン目で履歴（22度）を参照し、12ターン・660トークンまで崩れない** |

Qwen3 は `assistant_prefix` に空の `<think></think>` を指定する（Qwen3 の chat template が
`enable_thinking=False` のとき入れるブロック。**落とすと推論文を垂れ流す**）。

1.5B は `embed_table` / `pos0_kv` / `pin_pos0` を指定する（下の「1.5B のための拡張」）。
**会話がキャッシュ長（256）を超えると、画面に「古い発話から忘れます」と出る**。冒頭の会話は思い出せなくなるが、崩れはしない。

**★モデルを切り替えると会話はリセットされる**（2026-09-17）。セッションを開き直すので KV キャッシュは
失われるのに `chatStarted` が立ったままだと、**system prompt を含まない「2ターン目の差分」**を
新しいモデルへ渡してしまう（実機で `promptLen=22` になり、会話の途中から始まったように振る舞った）。

```
The capital city of Japan is Tokyo. The capital city of France is

窓16 固定    → Paris. ... is Washington, D. C. The capital of the United States is the
                                              ↑ 文脈が16トークンしか無く同じ話題に戻る
窓64 p-tensor→ Paris. The capital city of the United States is New York.     ← 誤り
窓64 p-chan  → Paris. The capital city of the United States is Washington.   ← 正しい
```

KV 画面（`kv_modes.json`）も同じ作りで、2つ並ぶ。

| ボタン | モデル | |
|---|---|---|
| KV decode **PER-CHANNEL** | `qwen_dec24pc_epctx.onnx` | 既定 |
| KV decode PER-TENSOR | `qwen_dec24_epctx.onnx` | 従来 |

```
Remember this fact: the access code is 8317. ... The access code is

per-tensor  → 8317. The answer is 8317. ... The answer was not learned by the
per-channel → 8317. The answer to the question "What is the access code?" is 8317.
```

どちらも `8317` は思い出せる（窓の外の情報を KVキャッシュが保持している）。
**per-channel の方が崩れにくい。**

> **設定ファイルが無ければモード行は出ない**（従来と同じ画面になる）。

リポジトリの [`jobs/demo_modes.json`](jobs/demo_modes.json) と
[`jobs/kv_modes.json`](jobs/kv_modes.json) が雛形である。

```bash
adb push jobs/demo_modes.json jobs/demo_override.json jobs/kv_modes.json jobs/kv_demo.json   /sdcard/Android/data/org.edgeslm.demo/files/jobs/
```

**モデル本体（772MB × 版の数）は別途 push が要る。** 現在の構成は4本:
`qwen24ext`（窓16）/ `qwen_pm64`（窓64 p-tensor）/ `qwen_pm64pc`（窓64 p-channel）/
`qwen_dec24`（KV p-tensor）/ `qwen_dec24pc`（KV p-channel）。

## 構成

```
demo_app/                  Android アプリ（android_runner からコピーして改変）
  app/src/main/cpp/
    runner/session.cpp       ORT/QNN セッション（コピー元のまま）
    runner/job.cpp           **task="generate" を追加**（窓ずらし生成）
    runner/progress.{h,cpp}  **新規**。生成の途中経過を UI へ渡す
    edgeslm_jni.cpp          nativeProgress() を追加
  app/src/main/java/org/edgeslm/demo/
    DemoActivity.kt          **新規**。デモ画面（レイアウトはコードで生成、XML/Compose なし）
    Qwen2Tokenizer.kt        **新規**。byte-level BPE（端末内トークナイズ）
    RunnerActivity/Service   検証用の入口（コピー元のまま。デモ画面とは独立）
host/                      トークナイザ資材の生成（uv プロジェクト。torch 不要）
jobs/                      検証用のジョブ定義
results/                   実測記録
```

**依存を1つも増やしていない**（ORT+QNN の AAR のみ）。QNN ランタイムと ORT の版が
噛み合っている現在の構成を崩さないため。

## 動かす

```bash
# 1) トークナイザ資材を作る（初回のみ）
cd host && uv sync && uv run python make_tokenizer_assets.py

# 2) ビルド & インストール
cd ../demo_app && JAVA_HOME=/d/Android/jdk17 ./gradlew --offline assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk

# 3) アプリを一度起動してディレクトリを作らせる（**アプリ自身に作らせる**必要がある）
adb shell am start -n org.edgeslm.demo/.DemoActivity

# 4) モデルと辞書を置く
S=/sdcard/Android/data/org.edgeslm.runner/files/models/ops
D=/sdcard/Android/data/org.edgeslm.demo/files
adb shell "cp $S/qwen24ext_epctx.onnx $S/qwen24_ctx.SM8350.bin $D/models/ops/"
adb push host/out/vocab.txt host/out/merges.txt host/out/specials.tsv host/out/golden.jsonl $D/tokenizer/
adb shell "chmod 644 $D/models/ops/* $D/tokenizer/*"
```

`--offline` を付けるのは、この環境では **JVM だけが外部 TCP に出られない**ため
（依存はキャッシュ済みなのでこれで通る）。

### トークナイザが HF と同じ ID 列を出すか確かめる

```bash
adb push jobs/tokenize.json /sdcard/Android/data/org.edgeslm.demo/files/jobs/
adb shell am start -n org.edgeslm.demo/.RunnerActivity   -a org.edgeslm.demo.RUN --es job jobs/tokenize.json
adb shell cat /sdcard/Android/data/org.edgeslm.demo/files/out/last_result.json
```

`{"ok":true,"total":31,"pass":31,"fail":0,"decode_total":31,"decode_pass":31}` が出れば
**端末側トークナイザは HF と一致している**。モデルは開かないので数秒で終わる。

照合ケースには**分解済み（NFD）の入力が混ざっている**。ここが落ちるときは
NFC 正規化が効いていない（→ [`results/D2_20260913_tokenizer_verify.md`](results/D2_20260913_tokenizer_verify.md)）。

### NPU に何本まで載るかを測る（task `memprobe`、2026-09-19 追加）

[EdgeSLM_AndroidScale](../EdgeSLM_AndroidScale)（1.5B 搭載の検証）の Phase 0-A のために足した。
指定したモデルを**1プロセスで順に開いて保持**し、RSS・ロード中の山（VmHWM）・dma-buf・MemFree を1段ごとに記録する。

- **モデルを開く前の状態から測る**ため、`nativeCreate` を通さず専用の JNI（`nativeMemProbe`）から入る（RunnerService で分岐）
- 途中で殺されても残るよう、`out/memprobe_<job_id>_steps.jsonl` / `_samples.csv` に1行ずつ追記する
- `hold_ms`（全部開いたまま待つ）/ `prereclaim_mb`（ロード前にメモリを少しずつ確保して解放し、MemFree を空けておく）を指定できる
- 駆動は PC 側の `EdgeSLM_AndroidScale/app/pipeline/memprobe.py`（記録・exit-info・logcat を回収する）

わかったこと（[P0A](../EdgeSLM_AndroidScale/results/P0A_20260919_memory_ceiling.md)）: 1プロセスの NPU 側上限は **dma-buf で約1,710MiB**。
ロード中は一時的に binary の約2倍を要求し、MemFree が少ないと **lmkd がホームアプリやキーボードまで kill し、最後にこのアプリも kill する**。
ただしこれは RunnerService（Foreground Service、oom_score_adj 200）から読んだときの話で、**画面が前面（adj 0）なら起きなかった**（[P3](../EdgeSLM_AndroidScale/results/P3_20260919_demo.md)）。

### 1.5B（EdgeSLM_AndroidScale）のための拡張（2026-09-19）

どれも **job の項目を書いたときだけ**効く。書かなければ従来どおり（既存のモードの動きは変えていない）。

| 項目 | どこで効くか | すること |
|---|---|---|
| `prereclaim_mb` | `Session::open`（全経路） | ロード前にメモリを少しずつ確保して解放し、MemFree を空けておく。**画面から読むときは不要**（成否は変わらずロードが約3.3秒延びるだけ。P3）。裏で読む計測経路（memprobe・RunnerService）向け |
| `embed_table` / `embed_dtype` | `generate_kv` | モデルの入力が `inputs_embeds` のとき、埋め込み表（既定 bf16 の [V][H]）を mmap して1行ずつ引く |
| `pos0_kv` | `generate_kv` | 会話の最初に、位置0（`<|im_start|>`）の K/V を**全スロットへ複製**して始める。1.5B は空きスロットを 0 にすると壊れる（[P1](../EdgeSLM_AndroidScale/results/P1_20260919_wiring_kcenter.md)） |
| `pin_pos0` | `generate_kv` | 会話がキャッシュ長を超えても**位置0 の K/V を窓に残す**（位置 L 以降の各ステップの前に slot 1 へ書き戻す）。無いと位置0 が押し出された直後に崩れる（1.5B で7ターン目に文字化け→以降は空の応答。P3 4章）。**`pos0_kv` が無いモデル（0.5B など）でも使える**（2026-09-19）: 1ターン目の最初のステップで端末が計算した位置0 の K/V を取っておき、持ち越す。0.5B KV512 も無いと 512 を超えた次のターンで崩れた（`uppiesque`→空の応答。[P8](../EdgeSLM_AndroidScale/results/P8_20260919_chat_ui.md)）。結果の `pin_pos0_src` は `file` / `captured` / `none` / `off` |
| `zero_copy` | `generate_kv` | **1ステップごとのコピーをやめる**（出力をこちらのバッファへ直接書かせ、KV は2組を入れ替える。バッファはセッションと同じ寿命）。回答は同じで、1.5B の12ターンでエンドツーエンドが 10〜14% 短い（[P6](../EdgeSLM_AndroidScale/results/P6_20260919_zero_copy.md)）。**KV 画面の3モードで既定**（`kv_modes.json`） |
| `htp_shared_io` | `Session::open` / `generate_kv`（`zero_copy` と併用） | ORT の QNN EP の HTP 共有メモリで KV などをコピー無しで NPU へ渡す。**この端末（Xperia 5 III）では使えない**: FastRPC に ORT 1.27 が必要とする `remote_register_buf_attr2` が無く、セッションを開けない（[P7](../EdgeSLM_AndroidScale/results/P7_20260919_htp_shared_memory.md)）。既定は無効 |
| `dialogs` | `generate_kv` | 独立した対話（`turns` の配列）を**1つのセッションで続けて**回す。対話ごとにキャッシュを捨てる。評価で問題ごとにモデルを読み直さないため（[P4](../EdgeSLM_AndroidScale/results/P4_20260919_eval.md)） |
| `ctx_retry_max` / `ctx_retry_wait_ms` | `Session::open`（全経路。既定 2 / 1500） | context の生成が **error 1002** で失敗したら、strict のまま待って開き直す。別のモデルを閉じた直後（Qwen3 → 1.5B の切り替え）に一時的に起きる。以前は非 strict へ進み、配置は同じなのに「一部 CPU」と表示されていた |

1.5B（28層・1,481.5MiB）は**デモの KV 画面のモード**に入れてある（上の表）。

**停止（キャンセル）したときもキャッシュを持ち越す**（2026-09-19）。持ち越すのは、それまでに実際にキャッシュへ入ったトークンまで
（結果の `carry_tokens`）。以前は止めると持ち越しを捨てていたため、画面が次のターンで差分だけを送ると、1.5B は
`pos0_kv needs prompt_ids[0] == <|im_start|>` で失敗し、0.5B は履歴なしで答えていた（KV 画面・チャット画面とも。[P8](../EdgeSLM_AndroidScale/results/P8_20260919_chat_ui.md)）。

KV 画面には、PC から同じ操作を繰り返して測るための**テスト用のインテント引数**がある（渡さなければ従来どおり）。
駆動は `EdgeSLM_AndroidScale/app/pipeline/ui_demo_test.py`（`chat` / `matrix` / `switch`）。

```bash
adb shell am start -n org.edgeslm.demo/.KvDemoActivity --ei mode 2 --ez chat true --es prompts "エアコンを22度にして|音楽をかけて|さっき何度にしたっけ？"
```

| 引数 | 意味 |
|---|---|
| `mode` | 起動時に開くモード（`kv_modes.json` の添字） |
| `chat` | 対話モードで始める |
| `prompts` | 各ターンが終わるたびに次の文を自動で送る（`\|` 区切り）。終わると会話を `out/kv_ui_chat.txt` に書き、logcat に `auto_chat done` を出す |

logcat（タグ `EdgeSLMKvDemo`）には `session ready` / `session failed` / `mode switched` / `chat answer` / `chat turn` / `turn stats` の1行が出る。

## 正直に言っておくこと（デモで聞かれたら）

| 質問 | 答え |
|---|---|
| 本当に端末だけ？ | **はい。** 機内モードで動く。モデルも辞書も端末内。トークナイズも端末（HF と ID 列 31/31 一致を端末で確認済み） |
| 文脈はどれくらい？ | 窓ずらし版は **64トークン**（最初の版は16）。KVキャッシュ版は充填したぶんすべて |
| 内容が事実と違うことがある | 0.5B の限界＋量子化誤差。per-channel 版でだいぶ減った（下記） |
| 量子化の劣化は？ | 実機 top-1 一致 **68.4%**（最終位置・窓64 per-channel）。per-tensor では 57.9% だった |
| なぜ per-channel で良くなる？ | 重み1本を1つの scale で表していたのを、出力チャンネルごとの scale にした。**速度は変わらない**（+0.7%） |
| KVキャッシュ版は？ | 動いている（17〜30 ms/token。プロンプト長と発熱で変わる）。`rms_norm` が原因ではなく、KVキャッシュの float 入力が量子化されない（C-1）のが原因だった |
| **Qwen3 は何が違う？** | 28層・872MB が**同じ端末の NPU で動く**（全ノード NPU、10.4 tok/s）。ただし**対話の中身は崩れる**（英語の思考文になる）。原因は位置0 の巨大活性に合わせた粗い量子化格子で、v68 の C-8 との組み合わせでは解けなかった（2026-09-17 クローズ）。詳細は [P14](../EdgeSLM_Android/results/P14_20260917_qwen3_closing.md)。**人に見せるときは Qwen2.5 で対話を、Qwen3 は「新しいモデルも載る」実演として使い分ける** |
| KV 版でも per-channel は効く？ | **生成は良くなるが、指標はほとんど動かない**（実機 top-1 60.4%→62.5%＝48件中1件）。ホストでは 68.8%→83.3% なので、**実機側に重み以外の支配要因がある**（KVキャッシュの float I/O を疑っている。未検証） |
| **1.5B は？** | 28層・1,481MiB が**全ノード NPU で約13 tok/s**。23問の比較で、**量子化と NPU で落ちた問題は無い**（FP32 と同等。[P4](../EdgeSLM_AndroidScale/results/P4_20260919_eval.md)）。間違えるのは 1.5B そのものの限界（本能寺の変を知らない等）。3ターン目で「さっき22度に設定しました」と履歴を参照でき、System Prompt も守る（0.5B はどちらもできなかった）。1プロセスの NPU 上限（約1,710MiB）の内側に収めるため、**埋め込み表（467MB）は NPU に載せず CPU 側で引いている**（[EdgeSLM_AndroidScale](../EdgeSLM_AndroidScale)） |
| 長く話し続けると？ | 1.5B はキャッシュ256トークン、0.5B は512トークン。超えると**古い発話から忘れる**が崩れない（画面に表示が出る。1.5B は24往復・1,522トークン、0.5B は24往復・822トークンまで確認）。先頭の `<|im_start|>` だけは残す作り（`pin_pos0`）で、残さないと超えた直後に崩れる（0.5B も同じ。P8）。**Qwen3 はこの対策を入れていない**（256 を超えたときの振る舞いは未確認） |

技術的な背景・制約（C-1〜C-12）はすべて
[EdgeSLM_Android/DoC/検証結果まとめ.md](../EdgeSLM_Android/DoC/検証結果まとめ.md) にある。
