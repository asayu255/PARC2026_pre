# 提出レポート

`PARC2026_report.docx` — PARC 2026 予選の提出レポート（A4 2 ページ以内、日本語）。
`build_report.js` が生成元で、`node build_report.js` で再生成できる
（`npm install docx` が必要。出力先はスクリプト末尾のパス）。

内容の根拠はすべて `PROGRESS.md` と `evidence/grading/` にある。

## 【要記入】は残り 1 か所

実測して本文へ入れ終わったもの:

| 項目 | 値 |
|---|---|
| 提出 zip の SHA-256 | `a2cc2f91ab7b77a423f65f306ef8d1fceccdd953df6c2c71128bff84088a37ea` |
| ベース重みの commit | `a85655ec941bae6644c9fbdf62db02b9726d7cf5` |

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

残る 1 項目は、このコンテナから Hugging Face への egress が塞がれているため
確認できていない。ブラウザで見て差し替えること。

### ベースモデルのライセンス

`https://huggingface.co/Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata`
のモデルカードに記載されたライセンス識別子を確認して差し替える。
基盤の OpenVLA / OpenVLA-OFT（`moojink/openvla-oft`）はいずれも MIT だが、
**この派生チェックポイント自身の表記は未確認である。**

## ページ数について

このコンテナでは LibreOffice が壊れていて（`.txt` すら変換できない）
レンダリングによる目視確認ができなかった。代わりに `word/document.xml` から
行数を積んで見積もっており、**約 1.14 ページ**（2 ページ制限に対して 43% の余裕）。
見積りの精度は ±15% 程度なので余裕は十分だが、**提出前に一度開いて
実際のページ数を確認すること。**
