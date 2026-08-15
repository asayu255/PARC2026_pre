# 提出レポート

`PARC2026_report.docx` — PARC 2026 予選の提出レポート（A4 2 ページ以内、日本語）。
`build_report.js` が生成元で、`node build_report.js` で再生成できる
（`npm install docx` が必要。出力先はスクリプト末尾のパス）。

内容の根拠はすべて `PROGRESS.md` と `evidence/grading/` にある。

## 【要記入】が 3 か所ある

このコンテナからは Hugging Face への egress が塞がれており、また提出 zip と
重みが GPU マシン側にあるため、**3 項目だけ実測できていない。**
GPU マシンで以下を実行して埋めること。

### 1. 提出チェックポイントのハッシュ値

提出したのは `tta5` 構成の zip。

```bash
sha256sum submission_oft_tta5.zip
sha256sum submission/model_weights/model-*.safetensors   # 重み本体も併記するなら
```

### 2. ベース重みの commit hash

`tools/fetch_model.sh` は `revision` を指定せずに `snapshot_download` を呼ぶので、
取得したのは **2026-08-11 時点の main の HEAD** である。まずローカルの
メタデータを見る:

```bash
ls -a submission/model_weights/.cache/huggingface/
cat submission/model_weights/.cache/huggingface/download/*.metadata 2>/dev/null | head
```

取れなければ Hub に問い合わせる。**ただしこれは「現在の」main の SHA** なので、
2026-08-11 以降にリポジトリが更新されていれば一致しない。その場合は
コミット履歴から当時の HEAD を特定すること。

```bash
python -c "from huggingface_hub import HfApi; \
  print(HfApi().model_info('Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata').sha)"
```

### 3. ベースモデルのライセンス

`https://huggingface.co/Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata`
のモデルカードに記載されたライセンス識別子を確認して差し替える。
基盤の OpenVLA / OpenVLA-OFT（`moojink/openvla-oft`）はいずれも MIT だが、
**この派生チェックポイント自身の表記は未確認である。**

## ページ数について

このコンテナでは LibreOffice が壊れていて（`.txt` すら変換できない）
レンダリングによる目視確認ができなかった。代わりに `word/document.xml` から
行数を積んで見積もっており、**約 1.13 ページ**（2 ページ制限に対して 43% の余裕）。
見積りの精度は ±15% 程度なので余裕は十分だが、**提出前に一度開いて
実際のページ数を確認すること。**
