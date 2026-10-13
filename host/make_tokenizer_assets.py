"""端末に置くトークナイザ資材を作る。

## なぜ vocab.json をそのまま置かないのか

端末側（Kotlin）で 151,643 件の JSON を毎回パースすると起動が遅くなる。
byte-level BPE のトークン文字列は**改行・タブ・空白を一切含まない**
（バイト 0x09/0x0A/0x20 はそれぞれ 'ĉ' 'Ċ' 'Ġ' のような表示可能文字に写像される）ので、
**1行1トークン・行番号＝ID** の素のテキストにできる。読み込みは readLines() 一発で済む。

出力（`out/` 配下、そのまま端末の files/tokenizer/ へ push する）:

    vocab.txt    151,643 行。行 i が ID i のトークン
    specials.tsv 追加トークン 22件。`ID<TAB>内容`（byte-level 変換を通さない生の文字列）
    merges.txt   BPE のマージ規則。先頭の #version コメントは落としてある

使い方:
    uv run python make_tokenizer_assets.py [--out out]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# 端末で照合するケース。**壊れやすい所を意図的に集めてある**。
#   - 短縮形は大文字小文字の両方（正規表現の (?i:) が効くか）
#   - 数字は1文字ずつ切れる規則（\p{N} が単数であること）
#   - 連続空白・行末空白・改行は \s+(?!\S) の後戻りが絡む
#   - 全角スペースは ICU の \s が Unicode 対応かどうかで変わる
#   - ★末尾5件は**分解済み（NFD）**。NFC 正規化が無いとここだけ落ちる
GOLDEN_CASES: list[tuple[str, str]] = [
    ('英語・基本', 'The capital city of France is'),
    ('英語・長め', 'The capital city of Japan is Tokyo. The capital city of France is'),
    ('短縮形', "It's John's car, they're here, I'd go, we've won, you'll see, I'm ok"),
    ('大文字短縮形', "IT'S JOHN'S CAR"),
    ('数字', 'Order 12345 costs $6,789.00 in 2026'),
    ('連続空白', 'a  b   c    d'),
    ('行末空白', 'trailing spaces   '),
    ('改行', 'line1\nline2\r\nline3\n\n\nline4'),
    ('タブ', 'col1\tcol2\t\tcol3'),
    ('記号', '!!! ??? ... --- *** (){}[]<>'),
    ('日本語', '日本の首都は東京です。'),
    ('日本語・長め', 'むかしむかし、あるところにおじいさんとおばあさんが住んでいました。'),
    ('全角スペース', '日本語\u3000全角スペース\u3000入り'),
    ('半角カナ', 'ﾊﾝｶｸ ｶﾀｶﾅ ﾃｽﾄ'),
    ('絵文字', 'Hello 👋 world 🌍 family 👨\u200d👩\u200d👧\u200d👦'),
    ('中国語', '中华人民共和国'),
    ('韓国語', '대한민국 서울특별시'),
    ('混在', 'Pythonで日本語を tokenize する 2026年'),
    ('特殊トークン', '<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n'),
    ('特殊トークン単体', '<|endoftext|>'),
    ('空白のみ', '   '),
    ('合成済みが', 'が ぎ ぐ げ ご'),
    ('合成済み é', 'café résumé naïve'),
    ('合成済みハングル', '한'),
    ('互換文字', '①②③ ㈱ ﬁ'),
    ('ゼロ幅', 'a\u200bb\u200cc'),
    ('NFD: 分解済みが', 'が ぎ ぐ げ ご'),
    ('NFD: 分解済み é', 'café résumé naïve'),
    ('NFD: ハングル分解', '한'),
    ('NFD: 結合記号', 'á̂̃b'),
    ('NFD: 混在', '日本語の が と café が混ざる'),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "out")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_id)
    src = Path(tok.vocab_file).parent if getattr(tok, "vocab_file", None) else None

    # tokenizer.json から直接取るのが確実（vocab.json / merges.txt が
    # スナップショットに無い場合もあるため）
    tj = json.loads(Path(tok.vocab_file).parent.joinpath("tokenizer.json").read_text("utf-8")) \
        if src and (src / "tokenizer.json").is_file() else None
    if tj is None:
        raise SystemExit("tokenizer.json が見つからない。HF キャッシュを確認すること")

    vocab: dict[str, int] = tj["model"]["vocab"]
    merges = tj["model"]["merges"]

    # ---- vocab.txt: 行番号 = ID ----
    n = max(vocab.values()) + 1
    table = [None] * n
    for t, i in vocab.items():
        table[i] = t
    missing = [i for i, t in enumerate(table) if t is None]
    if missing:
        raise SystemExit(f"ID に穴がある: {missing[:5]}...（{len(missing)}件）")
    # 改行を含むトークンがあれば1行1トークンの前提が壊れる。**黙って進めない**
    bad = [t for t in table if "\n" in t or "\r" in t]
    if bad:
        raise SystemExit(f"改行を含むトークンがある: {bad[:3]}")
    (args.out / "vocab.txt").write_text("\n".join(table), encoding="utf-8")

    # ---- specials.tsv: 追加トークン（byte-level 変換を通さない生文字列）----
    lines = []
    for a in tj.get("added_tokens", []):
        content = a["content"]
        if "\t" in content or "\n" in content:
            raise SystemExit(f"特殊トークンにタブ/改行: {content!r}")
        lines.append(f"{a['id']}\t{content}")
    (args.out / "specials.tsv").write_text("\n".join(lines), encoding="utf-8")

    # ---- merges.txt ----
    # tokenizer.json の merges は ["a b", ...] か [["a","b"], ...] のどちらもありうる
    mlines = []
    for m in merges:
        pair = m.split(" ") if isinstance(m, str) else list(m)
        if len(pair) != 2:
            raise SystemExit(f"マージ規則の形が想定外: {m!r}")
        mlines.append(f"{pair[0]} {pair[1]}")
    (args.out / "merges.txt").write_text("\n".join(mlines), encoding="utf-8")

    # ---- golden.jsonl: HF の ID 列。端末上で照合するための正解データ ----
    # **端末で測らないと意味がない**（ホストで同等アルゴリズムを動かしても、
    # 実際に動くのは Android の ICU 正規表現と java.text.Normalizer である）。
    # 分解済み（NFD）入力を**意図的に混ぜてある**——NFC 正規化の有無がここで出る。
    golden = []
    for name, text in GOLDEN_CASES:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        golden.append(json.dumps({"name": name, "text": text, "ids": ids},
                                 ensure_ascii=False))
    (args.out / "golden.jsonl").write_text(chr(10).join(golden) + chr(10), encoding="utf-8")

    print(f"vocab.txt    : {len(table):,} 行")
    print(f"specials.tsv : {len(lines):,} 件（ID {lines[0].split(chr(9))[0]}〜"
          f"{lines[-1].split(chr(9))[0]}）")
    print(f"merges.txt   : {len(mlines):,} 行")
    print(f"golden.jsonl : {len(golden):,} 件（端末で照合する正解 ID 列）")
    print(f"出力先       : {args.out.resolve()}")


if __name__ == "__main__":
    main()
