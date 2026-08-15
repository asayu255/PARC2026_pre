# 提出レポート

`PARC2026_report.docx` — PARC 2026 予選の提出レポート（A4 2 ページ以内、日本語）。
`build_report.js` が生成元で、`node build_report.js` で再生成できる
（`npm install docx` が必要。出力先はスクリプト末尾のパス）。

内容の根拠はすべて `PROGRESS.md` と `evidence/grading/` にある。

## 【要記入】は残っていない

実測して本文へ入れ終わったもの:

| 項目 | 値 |
|---|---|
| 提出 zip の SHA-256 | `a2cc2f91ab7b77a423f65f306ef8d1fceccdd953df6c2c71128bff84088a37ea` |
| ベース重みの commit | `a85655ec941bae6644c9fbdf62db02b9726d7cf5` |
| ベースモデルのライセンス | MIT License（著作権表示はテンプレートのまま未記入） |

commit は `.cache/huggingface/download/*.metadata` の 1 行目から復元した。
**全ファイルが同一の値を返した**ので、スナップショットが 1 コミットから
揃っていることも確認できている。

復元コマンド（`head -1` は複数ファイルだと `==> file <==` の見出しを付けるため、
`sort` に通すと見出しと中身が混ざる。1 ファイルずつ回すこと）:

```bash
for f in ~/parc_models/oft_libero_plus/.cache/huggingface/download/*.metadata; do
    head -1 "$f"
done | sort -u
```

## ページ数について

このコンテナでは LibreOffice が壊れていて（`.txt` すら変換できない）
レンダリングによる目視確認ができない。代わりに `estimate_pages.py` が
`word/document.xml` から高さを積んで見積もる。**約 1.95 ページ**（2 ページに
対して 3% の余裕）。

見積りを信用できる形にするため、build 側で 2 つ手当てをしてある:

- **行送りを `lineRule: EXACT` で固定した。** Yu Gothic は hhea の lineGap が
  大きく、`auto` のままだと 1 行が 1.5em 近くになる。フォント依存の一番大きい
  不確かさがこれで消え、残る誤差は折り返し数だけになった
- **半角文字の幅を 0.58em と多めに見積もっている**（`ASCII_EM`）。
  実際は小文字で 0.5em 前後なので、見積りは実寸より**厚めに出る**。
  制限を割る側へ間違えないための倒し方である

したがって 1.95 は上限側の値で、実際はこれより短くなる見込みだが、
**提出前に一度開いて実ページ数を確認すること。** 万一 3 ページ目に
こぼれていたら、§1.7「測定の作法と提出運用」の後半 1 段落を削るのが
いちばん影響が小さい。
