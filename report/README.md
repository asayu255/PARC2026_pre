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

## 書式の制約

- **本文・表とも最低 9pt**（`w:sz` を検査して 9pt 未満が 0 件であることを確認済み。
  9pt=104 箇所 / 9.5pt=6 / 11pt=4 / 13pt=1）
- **見出し下の横罫は廃止**（`pBdr` は 0 件）。表の枠線だけが残っている
- 行送りは `lineRule: EXACT`。本文 220 twip（1.22em）、表 205 twip（1.14em）。
  9pt を保ったまま 2 ページに収めるための圧縮で、ここを緩めると溢れる

## ページ数について

このコンテナでは LibreOffice が壊れていて（`.txt` すら変換できない）
レンダリングによる目視確認ができない。代わりに `estimate_pages.py` が
`word/document.xml` から高さを積んで見積もる。**約 1.92 ページ**（2 ページに
対して 4% の余裕）。

見積りを信用できる形にするため、build 側で 2 つ手当てをしてある:

- **行送りを `lineRule: EXACT` で固定した。** Yu Gothic は hhea の lineGap が
  大きく、`auto` のままだと 1 行が 1.5em 近くになる。フォント依存の一番大きい
  不確かさがこれで消え、残る誤差は折り返し数だけになった
- **半角文字の幅を 0.58em と多めに見積もっている**（`ASCII_EM`）。
  実際は小文字で 0.5em 前後なので、見積りは実寸より**厚めに出る**

したがって 1.92 は上限側の値だが、**提出前に一度開いて実ページ数を
確認すること。** 溢れていたら §1.3 の 2 段落目か §1.6 の最終段落を削るのが
いちばん影響が小さい。
