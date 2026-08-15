# 提出レポート

`PARC2026_report.docx` — PARC 2026 予選の提出レポート（A4 2 ページ以内、日本語）。
`build_report.js` が生成元で、`node build_report.js` で再生成できる
（`npm install docx` が必要。出力先はスクリプト末尾のパス）。

内容の根拠はすべて `PROGRESS.md` と `evidence/grading/` にある。

## 【要記入】は残り 2 か所

提出 zip の SHA-256 は実測済みで本文に入れてある:

    a2cc2f91ab7b77a423f65f306ef8d1fceccdd953df6c2c71128bff84088a37ea
    （submission_oft_tta5.zip）

このコンテナからは Hugging Face への egress が塞がれており、また提出 zip と
重みが GPU マシン側にあるため、**3 項目だけ実測できていない。**
GPU マシンで以下を実行して埋めること。

### 1. ベース重みの commit hash

**`submission/model_weights` を見ても出ない。** そこは SmolVLA 用で、
OFT+ の重みは `tools/stage_oft_submission.sh` の既定である
**`~/parc_models/oft_libero_plus`** にある（`PARC_OFT_WEIGHTS` で変更可）。

`tools/fetch_model.sh` は `revision` を指定せずに `snapshot_download` を呼ぶので、
取得したのは **2026-08-11 時点の main の HEAD** である。`local_dir` 付きで
落とすと 1 ファイルにつき `.metadata` が作られ、**その 1 行目が commit hash**:

```bash
head -1 ~/parc_models/oft_libero_plus/.cache/huggingface/download/*.metadata \
  | sort -u | head
```

これも空なら、HF のキャッシュ側にスナップショットが残っていないか見る
（`snapshots/` の直下のディレクトリ名がそのまま commit hash）:

```bash
ls ~/.cache/huggingface/hub/models--Sylvest--openvla-7b-oft-finetuned-libero-plus-mixdata/snapshots/
```

どちらも取れなければ Hub に問い合わせる。**ただしこれは「現在の」main の SHA**
なので、2026-08-11 以降にリポジトリが更新されていれば当時のものと一致しない。
その場合はコミット履歴から当時の HEAD を特定すること。

```bash
python -c "from huggingface_hub import HfApi; \
  print(HfApi().model_info('Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata').sha)"
```

### 2. ベースモデルのライセンス

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
