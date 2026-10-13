# D1: 端末内の生成ループ（native `generate` タスク）

| | |
|---|---|
| 実施日 | 2026-09-13 |
| 端末 | Sony Xperia 5 III SOG05（SM8350 / HTP **v68**）、unsigned PD |
| モデル | `qwen24ext_epctx.onnx` + `qwen24_ctx.SM8350.bin`（Qwen2.5-0.5B 全24層 / 772MB / `embed_mode=0`） |
| 配置 | `QNN HTP - EPContext - all nodes` / `strict_no_cpu_fallback: true`（**全ノード NPU**） |
| 目的 | 生成ループを**端末内**へ移し、①結果が検証済み経路と一致すること ②速度が実際に上がることを確かめる |

---

## 1. 結論

**同じ答えを、111倍速く出せるようになった。**

| | ホスト駆動（既存 `generate_on_device.py`） | **端末内ループ（本実装）** |
|---|---:|---:|
| 6トークン生成 | **28.8 s** | **0.26 s** |
| 1トークンあたり | 約 4,800 ms | **42.7 ms**（中央値） |
| うち OrtRun | 37 ms | **36.4 ms** |
| 生成結果 | ` Paris. The capital city of` | **完全一致** |

**推論そのもの（OrtRun 約36ms）は変わっていない。** 変わったのは呼び出し方だけである:

```
ホスト駆動 : 1トークンごとに adb push → am start → アプリ起動 → 1秒ポーリング → pull
端末内     : セッションを開いたまま session.run() を回す（ロードは1回きり）
```

**「37ms のモデルが体感1秒/トークンだった」のは、モデルではなく経路の問題だった**ことが
数字で確定した。

---

## 2. 一致の確認（これが実装の正しさの根拠）

窓の作り方と argmax の位置を `EdgeSLM_Android/app/pipeline/generate_on_device.py` と
**同一の規則**にしてある（右詰め・不足分は左を pad・見るのは常に位置 W-1・同点は先勝ち）。
そのうえで同じプロンプトを流し、トークン列が一致するかを見た。

```
プロンプト : The capital city of Japan is Tokyo. The capital city of France is
             （14トークン / 窓16）
期待       : [12095, 13, 576, 6722, 3283, 315]   ← fp32 と 6/6 一致まで確認済みの列
端末内実装 : [12095, 13, 576, 6722, 3283, 315]
一致       : ★完全一致
```

期待値は `EdgeSLM_Android/results/P4_generate_en.json`（fp32 と 6/6 一致を確認済み）。
**すでに fp32 まで遡って検証されている列と一致した**ので、この実装は経路を変えただけで
数値的な振る舞いを変えていない、と言える。

生データ: [`D1_gen_en.json`](D1_gen_en.json)

---

## 3. 長めの生成（デモの実際の見え方）

```
プロンプト : The capital city of Japan is Tokyo. The capital city of France is
出力(24トークン):
  " Paris. The capital city of the United States is Washington, D. C.
    The capital of the United States is the"

24トークン 0.98 秒 / 24.6 tok/s
1トークン 40.66 ms（中央値、実時間）— うち OrtRun 34.84 ms
```

窓16トークンの制約下でも**文として崩れていない**。デモに耐える。

生データ: [`D1_gen_long.json`](D1_gen_long.json)

---

## 4. 計測の取り方（数字を大きく見せないための約束）

**1トークンの時間として出すのは「実時間」であって OrtRun ではない。**

| 指標 | 何を含むか | 実測（中央値） |
|---|---|---:|
| `step_ms` | 推論 + logits の取り出し + argmax | **40.7〜42.7 ms** |
| `ort_run_ms` | OrtRun のみ | 34.8〜36.4 ms |

差の約6msは **logits `[1,16,151936]` float32 = 9.7MB のコピー**と、
最終位置 151,936 要素の argmax である。
`GenProgress` は両方を保持し、UI には `step_ms`（実時間）から出した tok/s を出す。
**OrtRun だけを見せると実際より速い数字になる**ため、そちらは内訳として扱う。

なお 9.7MB のコピーは `Session::run()` の出力取り出しがそのまま作っているもので、
最終位置の1行（607KB）だけ取れば省ける。**今回は最適化していない**——
6ms は 40ms に対して十分小さく、デモの成否を左右しないため。必要になれば削れる。

---

## 5. 実装

| 追加/変更 | 内容 |
|---|---|
| `cpp/runner/progress.{h,cpp}`（新規） | 生成の途中経過をロック付きで保持。UI は 50ms 間隔でポーリングする。JNI コールバックを避けた理由はヘッダの注記参照 |
| `cpp/runner/job.cpp` | `task == "generate"` を実装（`job.h` に v0.4 として枠が予約されていた箇所） |
| `cpp/edgeslm_jni.cpp` | `nativeProgress()` を追加 |
| `java/.../NativeBridge.kt` | 同上の宣言 |

`job.json` の指定:

```json
{ "task": "generate", "model": "models/ops/qwen24ext_epctx.onnx",
  "prompt_ids": [...], "max_new": 24, "pad_id": 151643, "stop_ids": [151645, 151643] }
```

`input_ids` 以外の入力（マスク入力版の `cos` / `sin` など）は `const_inputs` で
ファイルから読める。**モデルを差し替えても C++ を触らずに済む**ようにしてある。
