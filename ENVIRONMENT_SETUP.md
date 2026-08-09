# PARC 2026 現在の環境構築メモ

最終更新: 2026-08-09  
対象ホスト: `wakaba`  
リポジトリ: `/opt/home/ohara/PARC2026_pre`  
対象ブランチ: `claude/repository-code-progress-check-cconqv`

この文書は、管理者権限を使えない Ubuntu サーバー上で、PARC 2026 の Track 1 評価環境と SmolVLA ポリシー実行環境を構築した現在の状態をまとめたものである。

## 1. 現在の到達点

> このセクションは環境構築が終わった時点（2026-08-06）の記録である。
> その後の採点結果と方針は §22 以降が正しい。数字が食い違う場合は
> **後ろのセクションを採ること**（例: ここの「公開 4 タスクは 0%」は
> 配線の途中経過で、現在は 82.5%。§26）。最新の状況は §28〜§29。

### 完了しているもの

- Python 3.10 ベースの評価用 `venv`
- LIBERO-plus / LIBERO の取得
- LIBERO-plus の Track 1 スイート登録
- LIBERO アセットの取得と設定
- sudo なしでの OSMesa 導入
- ImageMagick / Wand の導入
- MuJoCo の OSMesa ヘッドレスレンダリング
- FastAPI ポリシーサーバーとの HTTP 通信
- `/health`、`/reset`、`/act` の疎通
- Track 1 のローカル評価と結果 JSON の保存
- GPU 用 `parc-policy` Conda 環境
- LeRobot 0.4.4 と CUDA 対応 PyTorch
- `lerobot/smolvla_libero_plus` の取得
- SmolVLA のオフライン GPU ロード
- `submission/policy_server.py` の `_load_model()` / `_predict_chunk()` 実装
- PARC 観測から LeRobot / SmolVLA 入力への変換（下記 16.1 で仕様を確定）
- 保存済み preprocessor / postprocessor を使った正規化・逆正規化
- SmolVLA による action chunk 推論（推論 0.24 秒、10 秒制限に対し 40 倍の余裕）
- 実モデルを使った Track 1 公開 4 タスクの評価

- 提出 ZIP 内だけで完結する VLM バックボーンの同梱（オフライン検証で証明済み）
- 提出 ZIP のビルド（`tools/make_submission.sh`）

- N_ACTION_EXEC の調整（50 -> 10）
- 公開 4 タスクでの評価: **総合 87.5%**（各 10 エピソード、追加学習なし）

| タスク | 難易度 | 成功率 | 平均ステップ |
|---|---|---|---|
| black bowl in top drawer -> plate | L3 | 80% | 131.8 |
| tomato sauce -> basket | L5 | 90% | 229.9 |
| milk -> basket | L2 | 100% | 153.6 |
| bowl -> stove | L4 | 80% | 90.2 |
| **総合** | | **87.5%** | |

40 エピソード・63 分を通してタイムアウトは発生しなかった。
レイテンシの実測は次のとおりで、制限 10 秒に対し最悪値でも 24 倍の余裕がある。

    遅い /act (>2s) の回数 : 0
    最悪値                 : 0.410s
    平均                   : 0.031-0.037s

- 採点環境と同条件での検証（tools/verify_clean_env.sh が PASS）

### 未完了のもの

- N_ACTION_EXEC を 10 より下げた場合の確認（5 / 2 / 1）
- 成功率のさらなる改善（追加学習）

### 採点環境に python3-dev が無い（1 回目の採点が 0 点になった原因）

採点環境では C 拡張をソースビルドできない。

    fatal error: Python.h: No such file or directory

このため lerobot は pip で入れられない。必須依存の pynput が Linux で
evdev を引き、evdev は wheel が一切公開されていない（sdist のみ）ため
必ずビルドが走って失敗する。推論の import 連鎖に pynput/evdev は
含まれておらず、lerobot が実機操作用に宣言しているだけである。

対策として、pip で入れられない純 Python のパッケージを同梱する。

| 同梱するもの | 理由 |
|---|---|
| `lerobot` | 上記のとおり pip で入れられない |
| `num2words` | transformers の SmolVLM プロセッサが要求する。lerobot は使わない |
| `docopt` | num2words の依存。wheel が無い |

`tools/vendor_lerobot.sh` が parc-policy からこれらをコピーし、
requirements.txt を「wheel のある依存だけ」に再生成する。dist-info も
一緒にコピーする（transformers の is_xxx_available() は
importlib.metadata.version() で有無を判定するため）。

`policy_server.py` は `_load_model()` で `vendor/` を sys.path の先頭へ挿す。

### 提出前の検証（必ず通すこと）

```bash
bash tools/vendor_lerobot.sh
bash tools/make_submission.sh
bash tools/verify_clean_env.sh
```

`verify_clean_env.sh` が採点環境を再現する。

| 再現する条件 | 方法 |
|---|---|
| まっさらな環境 | 新規 venv に requirements.txt だけを入れる |
| Python.h が無い | `pip install --only-binary=:all:`（ソースビルドを一切許さない） |
| HF キャッシュが無い | `HF_HOME` を空ディレクトリへ |
| サーバーもその環境で起動 | その venv の python で validate_submission.py を実行 |

作業環境で `validate_submission.py` を回すだけでは不十分である。必要な
ものが既に入っている状態での検証にしかならない。実際、それでは PASS
していたのに採点は 0 点だった。

この検証を入れてから、依存まわりの不具合を 4 回ローカルで検出した
（num2words の要否、pyserial / tqdm、gymnasium、num2words の再追加）。
採点の消費は 1 回で済んでいる。

### 注意: /act の 10 秒タイムアウトは実際に起きた

n_exec=50 の評価で /act が 10 秒を超え、トラック全体が 0 点になった。

    requests.exceptions.ReadTimeout: read timeout=10.0
    -> track1: 総合スコア 0.000

推論は実測 0.24 秒、スモークテストでも max 0.33 秒である。40 倍の外れ値で、
n_exec=25 / 10 は推論回数が 2 倍・5 倍多いのに起きていないため、系統的な
遅さではなく単発のストールと考えられる（GPU 競合・NFS・メモリ圧が候補）。

ただしクリーンな条件では再現していない。上記 4 タスク評価（40 エピソード）
では最悪値 0.410s、2 秒超はゼロだった。当時は複数のスイープが同時に走って
GPU を奪い合っていた時間帯であり、それが原因の可能性が高い。

成功率をいくら上げても、これが 1 回起きればトラックは 0 点になる。
policy_server.py は常時レイテンシを計測し、SLOW_REQUEST_SEC（既定 2 秒、
PARC_SLOW_SEC で変更可）を超えた /act をその場で警告し、エピソードごとに
n / mean / max / slow を出す。評価のたびにサーバーログを確認すること。

    grep -c '遅い /act' logs/server_full.log
    grep 'レイテンシ' logs/server_full.log

**提出可能な状態には到達している。** HF キャッシュを隠した状態
（`HF_HOME=/tmp/empty_hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`）で
`validate_submission.py submission.zip` が PASS することを確認済み。
zip は 2.1GB、推論レイテンシは mean 0.11 / max 0.33 秒（制限 10 秒）。

残るのは成功率で、公開 4 タスクは 0% である。配線は実測で全項目を
裏付けた（16.1）ので、残るのはモデルの能力の問題である。

### 外部通信への依存は 2 箇所あった

オフライン化で最も嵌まった点なので記録しておく。バックボーンを同梱して
`config.json` の `vlm_model_name` を直すだけでは**不十分**である。

| 箇所 | 値 | 対処 |
|---|---|---|
| `config.json` の `vlm_model_name` | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | `_load_model()` がローカルパスへ差し替え |
| `policy_preprocessor.json` の `tokenizer_processor.tokenizer_name` | 同上（**別経路**） | `preprocessor_overrides` で差し替え |

2 つ目は `AutoTokenizer.from_pretrained()` が独自にハブを見に行くもので、
`LocalEntryNotFoundError` で起動に失敗する。**HF キャッシュが見える状態では
再現しない**ため、`HF_HOME` を空にした検証でしか捕捉できない。

---

## 2. マシン情報

確認済みの実行環境は次のとおり。

```text
OS               Ubuntu 22.04.5 LTS
Kernel           6.8.0-58-generic
CPU architecture x86_64
GPU              NVIDIA RTX A6000
GPU memory       約 47.4 GiB / GPU
Repository       /opt/home/ohara/PARC2026_pre
Home             /opt/home/ohara
Miniforge        /opt/home/ohara/miniforge3
```

管理者権限を前提とせず、Python、ImageMagick、OSMesa をユーザー領域へ配置している。

---

## 3. Python 環境の分離

本構成では、評価環境とポリシー環境を分離している。

| 環境 | 用途 | 主な内容 |
|---|---|---|
| `PARC2026_pre/venv` | LIBERO 評価 | MuJoCo、robosuite、LIBERO-plus、評価パイプライン |
| `parc-venv-bootstrap` | 構築補助 | Python 3.10、virtualenv、ImageMagick |
| `parc-policy` | GPU ポリシー | LeRobot、SmolVLA、CUDA 対応 PyTorch、FastAPI |

### 重要な注意

評価用 `venv` と `parc-policy` を同じシェルで重ねて有効化しないこと。

誤った例:

```text
(parc-policy) (venv) ohara@wakaba:...
```

この状態では、表示上は `parc-policy` でも、実際には `venv` 側の Python が使われる場合がある。

解除してポリシー環境へ戻す:

```bash
deactivate
conda activate parc-policy
which python
which pip
```

期待値:

```text
/opt/home/ohara/miniforge3/envs/parc-policy/bin/python
/opt/home/ohara/miniforge3/envs/parc-policy/bin/pip
```

---

## 4. リポジトリ取得

```bash
cd ~
git clone <repository-url> PARC2026_pre
cd ~/PARC2026_pre
git checkout claude/task-overview-details-g4eu4t
```

現在の作業パス:

```text
/opt/home/ohara/PARC2026_pre
```

---

## 5. 評価環境の構築

### 5.1 通常のセットアップ

配布スクリプトの基本手順:

```bash
cd ~/PARC2026_pre
bash setup.sh
source env.sh
```

`setup.sh` は次を行う。

1. `venv` の作成
2. 評価用依存パッケージの導入
3. LIBERO-plus / LIBERO の clone
4. LIBERO-plus の既知パッチ
5. アセットの取得
6. `~/.libero/config.yaml` の生成
7. Track 1 スイート登録テスト

### 5.2 `python3.10-venv` が使えない場合

システム Python に `venv` モジュールがなく、sudo も使えない場合は、Conda 環境の Python と `virtualenv` を使う。

```bash
conda create -y -n parc-venv-bootstrap python=3.10 virtualenv
conda activate parc-venv-bootstrap

cd ~/PARC2026_pre
python -m virtualenv venv
bash setup.sh
```

現在は次の評価用環境が存在する。

```text
/opt/home/ohara/PARC2026_pre/venv
```

---

## 6. LIBERO / LIBERO-plus

配置先:

```text
/opt/home/ohara/PARC2026_pre/LIBERO-plus
/opt/home/ohara/PARC2026_pre/LIBERO
```

`setup.sh` により `~/.libero/config.yaml` が作成される。

現在の構成では概ね次を参照する。

```yaml
benchmark_root: /opt/home/ohara/PARC2026_pre/LIBERO-plus/libero/libero
bddl_files: /opt/home/ohara/PARC2026_pre/LIBERO-plus/libero/libero/bddl_files
init_states: /opt/home/ohara/PARC2026_pre/LIBERO-plus/libero/libero/init_files
datasets: /opt/home/ohara/PARC2026_pre/LIBERO-plus/libero/libero/datasets
assets: /opt/home/ohara/PARC2026_pre/LIBERO/libero/libero/assets
```

登録確認:

```bash
cd ~/PARC2026_pre
source activate_parc.sh

python - <<'PY'
import libero.libero.benchmark
from compe.t1 import register_t1

register_t1()
print("suite 登録 OK")
PY
```

---

## 7. sudo なしの OSMesa

システムへインストールせず、Ubuntu Jammy の `.deb` をユーザー領域へ展開した。

使用したパッケージ:

```text
libglapi-mesa  23.2.1-1ubuntu3.1~22.04.4
libosmesa6     23.2.1-1ubuntu3.1~22.04.4
libosmesa6-dev 23.2.1-1ubuntu3.1~22.04.4
libllvm15      1:15.0.7-0ubuntu0.22.04.3
```

展開先:

```text
$HOME/.local/parc-osmesa/root
```

展開例:

```bash
mkdir -p ~/.local/parc-osmesa/root

dpkg-deb -x libglapi-mesa_*.deb ~/.local/parc-osmesa/root
dpkg-deb -x libosmesa6_*.deb ~/.local/parc-osmesa/root
dpkg-deb -x libosmesa6-dev_*.deb ~/.local/parc-osmesa/root
dpkg-deb -x libllvm15_*.deb ~/.local/parc-osmesa/root
```

ライブラリディレクトリ:

```text
$HOME/.local/parc-osmesa/root/usr/lib/x86_64-linux-gnu
```

MuJoCo では次を設定する。

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
```

---

## 8. ImageMagick / Wand

ImageMagick は `parc-venv-bootstrap` Conda 環境へ導入している。

```bash
conda activate parc-venv-bootstrap
conda install -y imagemagick
```

配置先:

```text
/opt/home/ohara/miniforge3/envs/parc-venv-bootstrap
```

必要な環境変数:

```bash
export MAGICK_HOME="/opt/home/ohara/miniforge3/envs/parc-venv-bootstrap"
export LD_LIBRARY_PATH="$MAGICK_HOME/lib:${LD_LIBRARY_PATH:-}"
```

Wand 確認例:

```bash
python - <<'PY'
from wand.image import Image
print("Wand import OK")
PY
```

---

## 9. 評価用アクティベーションスクリプト

通常の評価では、リポジトリ直下の `activate_parc.sh` を使用する。

```bash
cd ~/PARC2026_pre
source activate_parc.sh
```

現在の `activate_parc.sh` は次をまとめて設定する。

- `env.sh` による評価用 `venv`
- `PYTHONPATH`
- `LIBERO_ROOT`
- ユーザー領域の OSMesa
- Conda 環境の ImageMagick
- `LD_LIBRARY_PATH`
- `MUJOCO_GL` / `PYOPENGL_PLATFORM`（既定 `egl`）

主な設定値:

```bash
export PARC_OSMESA_ROOT="$HOME/.local/parc-osmesa/root"
export PARC_OSMESA_LIB="$PARC_OSMESA_ROOT/usr/lib/x86_64-linux-gnu"
export MAGICK_HOME="/opt/home/ohara/miniforge3/envs/parc-venv-bootstrap"
export PARC_RENDERER="${PARC_RENDERER:-egl}"
export MUJOCO_GL="$PARC_RENDERER"
export PYOPENGL_PLATFORM="$PARC_RENDERER"
```

レンダラは `egl` が既定である（根拠は §24）。OSMesa に戻すには

```bash
PARC_RENDERER=osmesa source activate_parc.sh
```

`LD_LIBRARY_PATH` に `$PARC_OSMESA_LIB` を前置するのは `osmesa` のときだけである。
OSMesa の `libGL` は EGL のそれと衝突するため、egl では前置しない。

### curl の警告

`LD_LIBRARY_PATH` に Conda の `libcurl` が入るため、次の警告が出る場合がある。

```text
libcurl.so.4: no version information available
```

HTTP 応答が返っていれば評価への直接的な問題ではない。確認時は、必要に応じてクリーンな環境変数で curl を呼び出す。

```bash
env -u LD_LIBRARY_PATH /usr/bin/curl -fsS http://127.0.0.1:8002/health
```

---

## 10. 評価環境の動作確認

### 10.1 ポリシーサーバーの疎通

```bash
/usr/bin/curl -fsS http://127.0.0.1:8002/health
```

期待値:

```json
{"status":"ok"}
```

### 10.2 最小評価

評価用ターミナル:

```bash
cd ~/PARC2026_pre
source activate_parc.sh

python -m pipeline \
  --server-url http://127.0.0.1:8002 \
  --track track1 \
  --n-episodes 1 \
  --max-tasks 1 \
  --max-steps 20 \
  --timeout 10
```

確認済み事項:

- Track 1 のタスク読込み成功
- LIBERO 環境生成成功
- `/reset` と `/act` の通信成功
- 20 ステップ評価の完走
- 結果 JSON の保存

結果例:

```text
/opt/home/ohara/PARC2026_pre/results/server_8002.json
```

現在はゼロ action なので成功率 0% になる。

### 10.3 公開タスクのフル評価

実モデル接続後に実行する。

```bash
cd ~/PARC2026_pre
source activate_parc.sh

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$HOME/PARC2026_pre/results/full_${RUN_ID}"
LOG_FILE="$HOME/PARC2026_pre/logs/full_${RUN_ID}.log"

mkdir -p "$OUTPUT_DIR" "$HOME/PARC2026_pre/logs"
set -o pipefail

python -m pipeline \
  --server-url http://127.0.0.1:8002 \
  --track track1 \
  --n-episodes 2 \
  --max-steps 300 \
  --timeout 10 \
  --seed 42 \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_FILE"
```

`--verbose` はルートロガーを DEBUG にし、Numba の大量ログが出るため通常は付けない。

`--max-steps` は採点環境に合わせて **300** にする（根拠は §22）。
README の例が 600 だったため初期の計測は 600 で行っていたが、
その数字は採点結果より楽観的に出る。

---

## 11. ポリシー用 Conda 環境

作成例:

```bash
conda create -y -n parc-policy python=3.10 pip
conda activate parc-policy
```

ポリシーサーバーの基本依存:

```bash
cd ~/PARC2026_pre/submission
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

SmolVLA 用依存:

```bash
python -m pip install "lerobot[smolvla]" huggingface_hub
```

現在確認済みの主要バージョン:

```text
Python   3.10.20
LeRobot  0.4.4
PyTorch  2.10.0+cu128
CUDA     12.8
GPU      NVIDIA RTX A6000
```

確認コマンド:

```bash
conda activate parc-policy

python - <<'PY'
import sys
from importlib.metadata import version
import torch

print("Python:", sys.executable)
print("LeRobot:", version("lerobot"))
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA runtime:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

---

## 12. SmolVLA モデル

使用中の初期モデル:

```text
lerobot/smolvla_libero_plus
```

取得先:

```text
/opt/home/ohara/PARC2026_pre/submission/model_weights
```

取得コマンド:

```bash
cd ~/PARC2026_pre
conda activate parc-policy

hf download \
  lerobot/smolvla_libero_plus \
  --local-dir submission/model_weights
```

確認済みの主なファイル:

```text
submission/model_weights/README.md
submission/model_weights/config.json
submission/model_weights/model.safetensors
submission/model_weights/policy_preprocessor.json
submission/model_weights/policy_postprocessor.json
submission/model_weights/train_config.json
submission/model_weights/policy_preprocessor_step_5_normalizer_processor.safetensors
submission/model_weights/policy_postprocessor_step_0_unnormalizer_processor.safetensors
```

ローカル容量:

```text
約 1.1 GiB
model.safetensors は約 865 MiB
```

### 12.1 オフライン GPU ロード確認

```bash
cd ~/PARC2026_pre/submission
conda activate parc-policy

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 \
python - <<'PY'
import time
from pathlib import Path

import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

root = Path("model_weights").resolve()
device = torch.device("cuda:0")

print("Loading in offline mode...")
start = time.perf_counter()

model = SmolVLAPolicy.from_pretrained(
    str(root),
    local_files_only=True,
)
model = model.eval().to(device)

torch.cuda.synchronize()

print("Offline model load: OK")
print("Device:", next(model.parameters()).device)
print("Dtype:", next(model.parameters()).dtype)
print("Load time:", round(time.perf_counter() - start, 2), "seconds")
print("Chunk size:", model.config.chunk_size)
print("Action steps:", model.config.n_action_steps)
PY
```

確認済み結果:

```text
Offline model load: OK
Device: cuda:0
Dtype: torch.bfloat16
Load time: 6.92 seconds
Chunk size: 50
Action steps: 50
```

### 12.2 VLM バックボーンの注意

SmolVLA の初期化時には内部バックボーンとして次を参照する。

```text
HuggingFaceTB/SmolVLM2-500M-Video-Instruct
```

現在のサーバーでは Hugging Face キャッシュに取得済みのため、`HF_HUB_OFFLINE=1` でもロードできる。

ただし、本番評価環境にはこのユーザーキャッシュが存在しない。提出 ZIP を外部通信なしで起動するには、バックボーン一式を提出ディレクトリへ同梱し、`vlm_model_name` をそのローカルパスへ変更する必要がある。

取得例:

```bash
cd ~/PARC2026_pre/submission

hf download \
  HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --local-dir smolvlm_backbone
```

想定構成:

```text
submission/
├── policy_server.py
├── requirements.txt
├── model_weights/
│   ├── config.json
│   ├── model.safetensors
│   ├── policy_preprocessor.json
│   └── policy_postprocessor.json
└── smolvlm_backbone/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── ...
```

---

## 13. ポリシーサーバーの起動

### 同一マシン内で評価する場合

ポリシー用ターミナル:

```bash
# source activate_parc.sh は実行しない
conda activate parc-policy
cd ~/PARC2026_pre/submission

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 \
TOKENIZERS_PARALLELISM=false \
python policy_server.py \
  --host 127.0.0.1 \
  --port 8002
```

評価側の接続先:

```text
http://127.0.0.1:8002
```

### 別マシンから接続する場合

サーバー側:

```bash
python policy_server.py --host 0.0.0.0 --port 8002
```

クライアント側:

```text
http://<サーバー自身のIP>:8002
```

`--host` に別マシンの IP を指定してはいけない。その IP が現在のホストに割り当てられていない場合、次のエラーになる。

```text
cannot assign requested address
```

また、`--host 100.86.x.x:8002` のようにポートまで含めてはいけない。ホストとポートは別々に指定する。

---

## 14. ポートとプロセスの確認

```bash
ss -ltnp | grep ':8002'
pgrep -af 'policy_server.py|uvicorn'
```

プロセスの実行ディレクトリとコマンド確認:

```bash
PID=$(pgrep -f 'policy_server.py.*8002' | head -n 1)

echo "PID=$PID"
readlink -f /proc/$PID/cwd
tr '\0' ' ' < /proc/$PID/cmdline
echo
```

停止:

```bash
kill "$PID"
```

コードを変更した場合、起動中の Python プロセスは旧コードを保持しているため必ず再起動する。

---

## 15. 現在の MyPolicy 実装状態

> これも配線中の記録である。下の「未実装」は既に埋まっている
> （§16.1 で入出力を確定し、SmolVLA を接続済み）。action の選び方は
> §25.1（`N_ACTION_EXEC=5`）と §29（temporal ensembling）が現行である。

`submission/policy_server.py` には次の枠組みが実装済み。

- FastAPI サーバー
- `/health`
- `/reset`
- `/act`
- msgpack による観測のデシリアライズ
- float32 action のシリアライズ
- action queue
- action shape の `(7,)` への統一
- NaN / Inf の除去
- `[-1, 1]` への clip
- エピソードごとの queue リセット
- warmup の枠組み

未実装:

```python
def _load_model(self):
    # SmolVLA、preprocessor、postprocessor をロードする
    ...


def _predict_chunk(self, obs):
    # PARC 観測を SmolVLA 入力へ変換して推論する
    ...
```

現状では次を返す。

```python
np.zeros((ACTION_CHUNK_SIZE, 7), dtype=np.float32)
```

そのため、通信テストには使えるが性能評価には使えない。

---

## 16. PARC 観測と action 仕様

`POST /act` で受け取る主な観測:

```text
agentview_image           (128, 128, 3) uint8
robot0_eye_in_hand_image  (128, 128, 3) uint8
robot0_joint_pos          (7,) float
robot0_eef_pos            (3,) float
robot0_eef_quat           (4,) float
robot0_gripper_qpos       (2,) float
```

返却する action:

```text
shape: (7,)
dtype: float32
値域: [-1, 1]

[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

SmolVLA 接続時には次が必要。

1. 2 カメラ画像を LeRobot の期待するキーへ対応付ける
2. 画像を HWC uint8 からモデル前処理が扱える形式へ渡す
3. PARC のロボット状態から学習時と一致する state ベクトルを構成する
4. `self.instruction` を `task` として渡す
5. 保存済み preprocessor で正規化・トークナイズする
6. `(1, 50, 7)` の action chunk を推論する
7. postprocessor で action を逆正規化する
8. 必要な本数だけ queue へ入れる

### 16.1 確定した入出力仕様（実測で裏付け済み）

checkpoint の実物と LeRobot 0.4.4 のソースから確定させたもの。推測ではない。

| 項目 | 確定内容 | 根拠 |
|---|---|---|
| 入力キー | `observation.images.front` / `.wrist` | `policy_preprocessor.json` の rename_map が front→camera1、wrist→camera2 と定義。camera1 を直接渡すのは誤り |
| カメラ対応 | agentview→front、eye_in_hand→wrist | 同上 |
| 不足カメラ | camera3 / empty_camera_0 / 1 は渡さない | `prepare_images` が -1 埋めで補う (empty_cameras=2)。normalizer は存在キーのみ処理。学習時も front/wrist のみ |
| state 次元 | **8**（config の `[6]` ではない） | normalizer の統計が 8 次元で、`_apply_transform` は統計をスライスしない |
| state 構成 | eef_pos(3) + axis_angle(3) + gripper_qpos(2) | 統計の値域。実測 `max\|z\|`=1.52 で分布内 |
| quat 順序 | `(x, y, z, w)` | axis_angle の大成分が統計 mean と同じ 3 次元目に来る |
| 画像形式 | float32 CHW `[0,1]`。512 リサイズと `[-1,1]` 化はしない | `prepare_images` がモデル内部で実施。二重適用になる |
| 画像の向き | **180 度回転する** | A/B 実測。対象物体への最接近が 0.078 対 0.180 で 2.3 倍差 |
| chunk | 50 本、`postprocessor` で逆正規化 | config の chunk_size / n_action_steps |

計装は `PARC_DEBUG_DIR` を設定したときのみ有効になる。

```bash
PARC_DEBUG_DIR=/tmp/parc_dbg python policy_server.py --port 8002
```

生画像・実際にモデルへ入る画像・state の z-score・action chunk の統計を
最初の 3 回分ダンプし、毎ステップの手先と各物体の距離を `trace.csv` に記録する。
`PARC_FLIP180=0` で画像回転を無効化した対照群が取れる。

---

## 17. タイムアウト制約

- `/act`: 1 リクエスト 10 秒以内
- `/reset`: 1 リクエスト 10 秒以内
- サーバー起動: 既定 120 秒以内

平均ではなく、1 回でも 10 秒を超えるとトラック全体が error 扱いの 0 点になる。

SmolVLA のモデルロードは約 6.92 秒であり起動時間内に収まっているが、実際の action chunk 推論時間は別途測定する必要がある。

---

## 18. 提出前チェック

静的チェック:

```bash
cd ~/PARC2026_pre
source activate_parc.sh
python validate_submission.py submission.zip --static
```

動的チェック:

```bash
python validate_submission.py submission.zip
```

提出 ZIP の直下に必要なファイルを置く。

```text
submission.zip
├── policy_server.py
├── requirements.txt
├── model_weights/
└── 必要なら smolvlm_backbone/
```

ZIP 内を確認:

```bash
unzip -l submission.zip | head -n 100
```

誤り:

```text
submission/policy_server.py
```

正しい構造:

```text
policy_server.py
requirements.txt
model_weights/...
```

提出 ZIP の上限は 20 GB。採点環境では外部通信が使えないため、起動時の Hugging Face ダウンロードへ依存しないこと。

---

## 19. Git 管理上の注意

`venv/`、LIBERO-plus、結果ファイル等は `.gitignore` 対象である。

`submission/model_weights/` と `submission/smolvlm_backbone/` は巨大ファイルを含むため、通常の GitHub push へ含めないこと。GitHub の通常ファイルサイズ制限に抵触する可能性がある。

ローカルで誤って stage しないよう、必要に応じて `.gitignore` へ次を追加する。

```gitignore
submission/model_weights/
submission/smolvlm_backbone/
```

モデル重みは最終提出 ZIP には必要だが、通常のソースコードリポジトリへ入れる必要はない。

---

## 20. 推奨するターミナル構成

### ターミナル A: ポリシーサーバー

`tools/run_policy_server.sh` を使う。現在のシェルの状態に依存せず
parc-policy の python で起動するので、`(venv)` が残っていても構わない。

```bash
cd ~/PARC2026_pre
bash tools/run_policy_server.sh
```

このスクリプトが行うこと:

- parc-policy の python を探して使う（`PARC_POLICY_PYTHON` で明示指定も可）
- lerobot が入っていなければ起動前に明確なエラーで止める
- `VIRTUAL_ENV` / `PYTHONPATH` / `LD_LIBRARY_PATH` を引き継がない。
  `activate_parc.sh` は PYTHONPATH に LIBERO-plus を、LD_LIBRARY_PATH に
  OSMesa と ImageMagick を入れるが、これらはポリシー側には不要で
  conda 環境のライブラリと競合しうる
- `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` / `TOKENIZERS_PARALLELISM` /
  `CUDA_VISIBLE_DEVICES` の既定を設定する

実験用のつまみ:

```bash
PARC_N_EXEC=25   bash tools/run_policy_server.sh   # open-loop 区間の長さ
PARC_FLIP180=0   bash tools/run_policy_server.sh   # 画像 180 度回転を無効化
PARC_DEBUG_DIR=/tmp/parc_dbg bash tools/run_policy_server.sh   # 計装を有効化
PARC_OFFLINE_TEST=1 bash tools/run_policy_server.sh  # HF キャッシュを隠す
PARC_PORT=8003   bash tools/run_policy_server.sh
```

### ターミナル B: 評価パイプライン

```bash
cd ~/PARC2026_pre
source activate_parc.sh

python -m pipeline \
  --server-url http://127.0.0.1:8002 \
  --track track1 \
  --n-episodes 1 \
  --max-tasks 1 \
  --max-steps 20 \
  --timeout 10
```

### ターミナル C: GPU・プロセス監視

```bash
watch -n 1 nvidia-smi
```

または:

```bash
ss -ltnp | grep ':8002'
pgrep -af 'policy_server.py|uvicorn'
```

---

## 21. 次の作業

配線は完了し実測で裏付けた。残りは 2 つである。

### A. 提出可能な状態の維持（完了）

`tools/make_submission.sh` が zip を作る。不要物（`__pycache__` / `.cache` /
`onnx/` / `eval/` の動画 / `*.gguf` 等）を落とし、zip 直下が
`policy_server.py` になる構成で固める。backbone は onnx 除去で
5.88GB から 1.9GB になった。

```bash
bash tools/make_submission.sh
HF_HOME=/tmp/empty_hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /opt/home/ohara/miniforge3/envs/parc-policy/bin/python validate_submission.py submission.zip
```

`validate_submission.py` はサーバーを `sys.executable` で起動する（:554）。
評価用 venv の python から実行すると lerobot が見つからず失敗するので、
parc-policy の python をフルパスで指定すること。

### B. 成功率の改善（追加学習）

ベースモデルは対象物体まで到達できるが把持に至らない（最接近 7.8cm の後に離脱）。
要因は次の 2 つと考えられる。

- 学習は 256x256、PARC は 128x128（`pipeline/config.py:51`）。画素数が 1/4 で、
  これは評価側の固定値なので提出側では回復できない
- ベースモデルの学習タスクは 40 種類のみ（`task_index.max=39`）。PARC の 4 タスクは
  背景テクスチャ・照明を変えた L2〜L5 の摂動版である

`examples/smolvla_libero_spatial_lora.ipynb` を出発点に LoRA 追加学習を行う。
ノートブックは 10 タスク x 5 エピソード・3000 steps の最小構成なので、
PARC のタスク構成に近いデータで条件を組み直す。

---

## 22. 採点環境から判明した事実

2 回目の提出（提出 ID `server_8000`、合計 192.7 秒、track1 総合スコア 0.043）の
採点ログから読み取れた事実を記録する。ローカルの計測条件をこれに合わせること。

### 22.1 エピソード長は 300 ステップ

採点ログの latency 行は `n=300 n=300 n=300 n=300 n=216 n=161 n=300` で、
300 で頭打ちになっている。`pipeline/cli.py:65` の `--max-steps` の既定値が
300 であり、採点ハーネスはこれを上書きしていない。

ローカルで 600 ステップ・公開 4 タスク・各 10 エピソードで測った 87.5%
（80 / 90 / 100 / 80）は、この点で楽観的である。例えば
`pick_up_the_tomato_sauce_and_place_it_in_the_basket_table_27` は
成功までの平均が **229.9** ステップで、300 打ち切りだと取りこぼしが増える。
以降の計測・スイープはすべて `--max-steps 300` で行う。

なお `n=216` と `n=161` の 2 本は 300 未満で終わっている。これは `done` が
立ったということであり、ポリシーはゴールに到達できている。

### 22.2 `3_Omni` はタスクセット名ではない

ログの `[evaluate] タスク用途: 3_Omni` の `Omni` は **OmniCampus**、
すなわち提出プラットフォームのことである。このリポジトリの `evaluate.py:170`
に `build_omnicampus_result()` があり、採点結果は OmniCampus 形式の JSON で
返される。`3_Omni` は OmniCampus 側の課題スロットのラベルであって、
LIBERO のタスクスイート名ではない。この行は「どのタスクで採点されたか」を
何も語っていない。

なお `[evaluate] タスク用途:` という行自体はこのリポジトリのどこにも無く、
プラットフォーム側のラッパーが出している。

採点ログにはパイプライン自身の `[INFO] pipeline.*` 行が**含まれていない**。
`evaluate.py:164` は子プロセスの stdout/stderr を親の stderr に流しており、
実際 robosuite や gym の警告は出ているので、出力が捨てられているのではなく
採点環境のログレベルが WARNING 以上になっている。したがって
`タスク評価完了: <名前> — 成功率 X%` は採点ログから読めず、
**どのタスクで採点されたかを知る手段は無い。**

### 22.3 採点は 8 エピソードで、成功は 0 本である

採点ログの `POST /reset` は 8 回である。`reset()` は「前エピソードの」
レイテンシを出すので latency 行は 7 本（最終エピソード分は出ない）。
内訳は n=300 が 6 本、n=216 と n=161 が各 1 本。

ここから 0.043 の性質が決まる。`overall_score` が
`np.mean([ts.success_rate for ts in task_scores])`（`pipeline/scorer.py:102,130`）
だとすると、8 エピソードをどうタスクに分割しても取り得る値は
1/8、1/7、1/(2·7) … といった離散値にしかならず、**0.043 は作れない**。
最小の非零値でも 0.0714（7 タスク中 1 タスクが 1/2）である。

したがって採点側の `overall_score` は配布版 `scorer.py` の定義とは違い、
配布されていない `total_score_config.json` を使った重み付き複合値である。
そして 0.043 という小ささは、**成功が 0 本で、衝突・軌跡系の補助指標
（collision / cartesian / jerk / sparc）の部分点だけが残った状態**と整合する。

### 22.3.1 2 本はゴールに到達していて、それでも失敗している

`pipeline/rollout.py:236` はエピソードを `done` でしか打ち切らない。
n=216 と n=161 の 2 本は 300 未満で終わっているので、`done` は立っている。
一方で成功が 0 本なので、この 2 本は
`success = bool(done) and not collided`（:241-242）の後半で落ちている。
つまり **非対象物体を 1 mm 超動かした衝突失敗**である。

これは重要な切り分けである。採点セットでの内訳は

- 6/8 … ゴールに到達できず 300 step 打ち切り
- 2/8 … ゴールに到達したが衝突で失格

ローカルの公開 4 タスク（§23）では 300 step 到達がほとんど無く、
stove では 10/10 が `done` に達していた。採点セットのほうが明らかに難しい。

### 22.3.2 以前の誤った読み

このセクションは 2 度書き換えている。経緯を残す。

1. 「重み付き複合値」— 結論としては正しかったが、根拠が甘かった
2. 「成功率の単純平均」— `scorer.py` の配布版コードだけを見て断定した誤り。
   採点側が同じコードを使っているという前提が誤っていた

配布版の `scorer.py` は採点側の実装ではない。ローカルの `総合スコア` と
採点の `総合スコア` は同名だが別物として扱うこと。

### 22.4 レンダリングは EGL（GPU）

採点環境は EGL / NVIDIA でレンダリングしており、ローカルの OSMesa
（ソフトウェアレンダリング）より約 4 倍速い。合計 192.7 秒で完走している。
`/act` のレイテンシは mean 0.042s / max 0.455s / slow=0 で、
10 秒制約に対しては十分な余裕がある。この余裕は N_ACTION_EXEC を
小さく（=推論回数を増やす）する方向の予算として使える。

---

## 23. 300 ステップでの基準値

公開 4 タスク・各 10 エピソード・`--seed 42`・`N_ACTION_EXEC=10`。
所要 3348 秒（OSMesa）。

| タスク | 600 step | 300 step | 成功時の平均 step | 失敗の性質 |
|---|---|---|---|---|
| pick_up_the_black_bowl_in_the_top_drawer_..._table_2 | 80.0% | 80.0% | 131.6 | 2 本が 300 打ち切り |
| pick_up_the_tomato_sauce_..._table_27 | 90.0% | 100.0% | 193.5 | なし |
| pick_up_the_milk_..._light_15 | 100.0% | 70.0% | 154.0 | 3 本が 300 打ち切り |
| put_the_bowl_on_the_stove_light_11 | 80.0% | 80.0% | 89.6 | 2 本が衝突失敗 |
| overall | 87.5% | **82.5%** | | |

`avg_steps` は成功エピソードのみの平均である（`pipeline/rollout.py:56-61`）。

### 23.1 失敗の 2 分類

`rollout.py:236` はエピソードを `done` でしか打ち切らない。したがって
300 未満で終わったエピソードは必ず `done` が立っている。それでも失敗なら、
`success = bool(done) and not collided`（:241-242）の後半に引っかかった、
つまり非対象物体を 1 mm 超動かしたということになる。ログの
`[進捗] ... 50/300` の再開位置を数えれば、どちらの失敗かを切り分けられる。

- **時間切れ失敗**（milk / drawer）: 300 まで走り切って `done` が立たない。
  milk は 600 なら 100% なので、この 3 本は「あと少し」のケース。
- **衝突失敗**（stove）: 10 本すべてが平均 90 step 前後で `done` に達しており、
  失敗 2 本はステップ数ではなく軌道の粗さが原因。ステップを増やしても改善しない。

600 → 300 で落ちた 5 pt の実体は milk の 3 本ぶんで、残りは
flow-matching のサンプリング由来のばらつき（10 エピソードだと ±10 pt は動く）。
300 打ち切りそのものの影響は限定的である。

### 23.2 次に効きそうな順序

衝突失敗は追加学習なしで触れる余地がある。1 mm というしきい値は極端に厳しいので、
`N_ACTION_EXEC` を下げて閉ループ性を上げると軌道が滑らかになる可能性がある。
レイテンシ予算（§22.4）は十分余っている。

```bash
PARC_SWEEP_MAX_STEPS=300 PARC_SWEEP_NEXEC="10 5 2" \
  nohup bash tools/sweep_nexec.sh > logs/sweep300.log 2>&1 &
```

時間切れ失敗のほうは追加学習（§21 B）の領域。

### 23.3 N_ACTION_EXEC スイープの結論（この軸は閉じた）

`put_the_bowl_on_the_stove_light_11`・10 エピソード・300 step・seed 42。

| n_exec | success | collision | cartesian | jerk(rms) | sparc |
|---|---|---|---|---|---|
| 10 | 0.70 | 0.300 | 0.800 | 6.477 | -2.316 |
| 5 | 0.80 | 0.200 | 0.814 | 6.140 | -2.322 |
| 2 | 0.80 | 0.200 | 0.815 | 7.121 | -2.306 |

**成功率の差はノイズである。** 根拠は表の中ではなく対照条件のずれにある。
`n_exec=10` は §23 のフル評価の stove と完全に同一設定だが、
あちらは 80.0%、このスイープでは 70.0% だった。同じ設定で 10 pt 動いている。
10 エピソード・p≈0.8 の二項標本の標準誤差は √(0.8×0.2/10) = 12.6 pt なので、
観測された 10 pt 差は 0.8σ 未満にすぎない。`collision` の 0.300 → 0.200 も
1 エピソードぶんである。

信用できるのは `jerk(rms)` だけである。10 エピソード × 約 90 step ≒ 900 サンプルの
平均なので成功率より分散が小さい。ここだけ非単調で 5 が最小、2 が最悪（+16%）。
open-loop を短くするほどチャンク境界での再計画が増え、flow-matching は毎回
ノイズを引き直すので境界で不連続が出る。逆に長すぎると誤差が溜まって補正が跳ねる。
中間に最小値ができるのは機構として自然である。

ただしこれは **`n_exec=2` を避ける根拠**にはなっても、10 と 5 の差（5%）は小さく、
既定値 10 を動かす根拠にはならない。**`N_ACTION_EXEC = 10` を維持する。**

> **この判断は §25.1 で覆した。** EGL で 50 エピソードまで本数を増やしたところ、
> ここで唯一信用できるとした jerk の順序がそのまま再現し、success と collision
> も同じ向きを指した。既定は 5 にした。このセクションは「10 エピソードでは
> 判断できなかった」という記録として残す。

決着させようとすると割に合わない。p≈0.75 付近で 10 pt 差を有意に検出するには
1 条件あたり 300 エピソード程度が要る。stove は 10 エピソードで約 8 分なので
1 条件 4 時間、2 条件で 8 時間になる。チューニングパラメータ 1 個に払う額ではない。

10 エピソードのスイープで読めるのは「大きく壊れていないか」までである。
今後この種の A/B を回すときは、成功率ではなく `jerk` / `cartesian` のような
連続量を主指標に置くこと。

### 23.4 スイープ実行時の注意

同じコマンドを 2 回貼ると 2 つ起動する。flock（`tools/sweep_nexec.sh:54-66`）が
後発を `Exit 1` で止めるので評価自体は壊れないが、リダイレクト
（`> logs/sweep300.log`）はスクリプト実行前に効くのでログは truncate される。
先行プロセスは自分のファイルオフセットのまま書き続けるため、
ログ先頭が NUL 埋めの穴になる。ログ名に `$$` を入れておくと安全である。

```bash
nohup bash tools/sweep_nexec.sh > "logs/sweep300_$$.log" 2>&1 &
```

条件ごとの `logs/eval_nexec<N>.log` / `logs/server_nexec<N>.log` と
`results/nexec<N>/*.json` は別ファイルなので、この事故では失われない。

---

## 24. EGL 検証の結果: レンダラは原因ではない。ただし 7.7 倍速い

採点は 8 エピソードで成功 0 本（§22.3）、ローカルの公開 4 タスクは
300 step で 82.5%（§23）。この落差はモデルの弱さでは出ない大きさなので、
まず環境差のうちいちばん大きいレンダラ（採点 EGL / ローカル OSMesa）を疑い、
ローカルを EGL に切り替えて同条件で回した。

### 24.1 成功率: 区別できない

公開 4 タスク × 2 エピソード（採点の 8 本に合わせた本数）、300 step、seed 42。

| レンダラ | エピソード数 | 成功率 |
|---|---|---|
| OSMesa | 40（4 タスク × 10） | 82.5% |
| **EGL** | 8（4 タスク × 2） | **62.5%**（100 / 50 / 50 / 50） |
| 採点 | 8 | 成功 0 本 |

2 エピソードだとタスクごとの値は 0 / 50 / 100 しか取り得ず、標準誤差は
√(0.8×0.2/2) = 28 pt ある。62.5% と 82.5% は区別できない。
**EGL でポリシーが壊れることはない。** レンダラ仮説は否定された。

したがって採点で成功 0 本になった原因は、**採点タスクセットが公開 4 タスクとは
別物で、かつ大幅に難しい**ことに帰着する。EGL で同じ 8 本構成を回して 5/8 成功する
一方、採点は 0/8 である。

### 24.2 EGL は 1 エピソードあたり 7.7 倍速い

| レンダラ | 1 エピソードあたり |
|---|---|
| OSMesa | 83.7 秒（3348s / 40ep） |
| EGL | **10.9 秒**（87.2s / 8ep） |

これが今回いちばん大きい収穫である。測定コストの前提が変わる。
§23.3 で「n_exec の差を有意に検出するには 1 条件 300 エピソード = 4 時間、
2 条件で 8 時間」として諦めた計算が、**1 条件 55 分**になる。
衝突率も実測で詰められる本数が回せるようになった。

`activate_parc.sh` の既定を `egl` にした。OSMesa に戻すには
`PARC_RENDERER=osmesa source activate_parc.sh`。

なお採点環境は 8 エピソードで 192.7 秒（24 秒/ep）で、ローカル EGL より遅い。
サーバー起動待ち 20 秒とモデルロードが含まれ、かつ採点側は 6/8 が
300 step 完走している（ローカルは早期終了が多い）ためで、矛盾しない。

---

## 25. 残っている作業

汎化以外の逃げ道が無くなった。優先順に 2 つ。

### 25.1 衝突失敗を減らす（EGL で本数を稼いで測る）

採点セットで到達できた 2 本は、どちらも 1 mm ルールで失格している（§22.3.1）。
到達しても衝突すれば 0 点なので、ここは汎化とは独立に効く。

EGL になったので、成功率ではなく**衝突率**を主指標にして本数を稼げる。

```bash
PARC_SWEEP_MAX_STEPS=300 PARC_SWEEP_EPISODES=50 PARC_SWEEP_NEXEC="10 5 2" \
  nohup bash tools/sweep_nexec.sh > "logs/sweep_egl_$$.log" 2>&1 &
```

50 エピソードなら衝突率の標準誤差は √(0.2×0.8/50) = 5.7 pt で、
§23.3 の 10 エピソード（12.6 pt）とは判断できる範囲が違う。

#### 結果: n_exec=5 を採用する（§23.3 の結論を覆す）

EGL・stove・50 エピソード・300 step・seed 42。1 条件 約 5.5 分。

| n_exec | success | collision | cartesian | jerk(rms) | sparc |
|---|---|---|---|---|---|
| 10 | 0.84 | 0.120 | 0.842 | 6.470 | -2.421 |
| **5** | **0.92** | **0.080** | 0.838 | **6.091** | -2.371 |
| 2 | 0.80 | 0.180 | 0.826 | 6.899 | -2.407 |

まず、10 エピソードで 70〜80 に振れていた `n_exec=10` が 50 本で 84.0% に
落ち着いた（SE 5.2 pt）。これが基準値である。

success の 0.92 vs 0.84 は差 8 pt に対し SE 6.5 pt、1.24σ で単独では有意でない。
**決め手は jerk が独立な 2 回で同じ順序を再現したこと**である。

| n_exec | §23.3（OSMesa 10ep） | 今回（EGL 50ep） |
|---|---|---|
| 10 | 6.477 | 6.470 |
| 5 | 6.140 | 6.091 |
| 2 | 7.121 | 6.899 |

レンダラもエピソード数も違うのに値がほぼ一致し、5 が最小・2 が最悪という
順序が保たれている。jerk は 1 エピソード約 90 step × 本数の平均で分散が
小さい（§23.3 でも「信用できるのは jerk だけ」と書いた）。その再現の上で、
今回は success と collision も同じ向きを指した。3 指標の一致を採用根拠とする。

採用の決め手をもう一つ。**collision が 0.120 → 0.080** である。採点で
ゴールに到達した 2 本はどちらも 1 mm ルールで落ちている（§22.3.1）ので、
ここは採点スコアに直接効く。レイテンシは採点実測で max 0.455 秒（制限 10 秒）
なので、推論回数が 2 倍になってもコストは無い。

`submission/policy_server.py` の `N_ACTION_EXEC` の既定を 10 から **5** に変更した。

`n_exec=2` が悪化するのは機構としても筋が通る。短くするほどチャンク境界での
再計画が増え、flow-matching は毎回ノイズを引き直すので境界で不連続が出る。
逆に長すぎると誤差が溜まって補正が跳ねる。中間に最小ができる。

### 25.2 LoRA 追加学習: データ構成の設計

出発点は `examples/smolvla_libero_spatial_lora.ipynb`。仕組みは確認済みで、
`lerobot/libero_plus` データセット（revision 固定）からタスク名で
エピソードを選び、`lerobot-train --dataset.episodes=[...]` に渡す。
LoRA は `--peft.method_type=LORA`、マージは `PeftModel.merge_and_unload()`。

#### 設計の根拠

`compe/t1/T1_TASKS.csv` が採点セットの性質を語っている。公開 4 タスクは

| タスク | 親スイート | 摂動カテゴリ | 難度 | libero_plus_id |
|---|---|---|---|---|
| bowl_in_top_drawer_table_2 | libero_spatial | Background Textures | L3 | 80 |
| tomato_sauce_table_27 | libero_object | Background Textures | L5 | 241 |
| milk_light_15 | libero_object | Light Conditions | L2 | 2408 |
| bowl_on_stove_light_11 | libero_goal | Light Conditions | L4 | 2467 |

- 親スイートが 3 種にまたがる → 採点も全スイートに及ぶと考えるべき
- `libero_plus_id` が 2467 まである → 採点プールは LIBERO-plus の
  摂動バリアント空間（40 親タスク × 摂動の数千通り）
- ベースモデルは 40 親タスクを知っている（`task_index.max=39`）のに
  摂動版で失敗する → 足りないのはタスク知識ではなく**摂動耐性**

デモデータは親タスク（無摂動）のものしか無いので、摂動耐性は
**データの広さ**と**画像オーグメンテーション**で作るしかない。

#### データセットの実測（`tools/inspect_dataset.py`）

```text
codebase_version: v3.0   total_episodes: 14347
total_frames: 2238036    total_tasks: 40    fps: 20
length: 平均 156.0  中央 138  最小 75  最大 505
タスクあたり: 最小 146  中央 385  最大 500
```

`episodes` メタデータに**摂動を識別する列は無い**。列は `episode_index` /
動画ポインタ / `tasks` / `length` と各特徴量の `stats` だけである。
したがってメタデータからの層別サンプリングはできない。

ただし `stats/observation.images.front/mean` から摂動の有無は判定できる。
背景テクスチャと照明の摂動は画像の平均輝度を動かすからである。実測:

| 指標 | 値 |
|---|---|
| 全体の平均輝度 | 0.4136（std 0.1201） |
| 輝度の範囲 | **0.0856 〜 0.7231（8 倍）** |
| タスク内 std の平均 | **0.1029**（全体 std の 86%） |

ばらつきの 86% がタスク内で生じている。**摂動バリアントは実データに
含まれている。** 1 タスク約 385 本という本数も、元の LIBERO の
50 デモ/タスクに対して約 7.7 倍で、摂動条件 7〜8 種 × 50 デモという
構成と整合する。

この結果で設計方針が変わる。**オーグメンテーションで摂動を人工的に作る
必要は薄く、実データを広く取ることが摂動網羅に直結する。**
`image_transforms` は「学習カタログの外側の条件」への保険として残すが、
主役ではない。

#### 決定した構成（ノートブックとの差分）

| 項目 | ノートブック | 今回 | 理由 |
|---|---|---|---|
| タスク | spatial 10 | **全 40 親タスク** | 採点は全スイートに及ぶ |
| エピソード/タスク | 5 | **60**（均等間引き） | 計 2,400 本 ≒ 374k frame。摂動条件 7〜8 種に各 7〜8 本当たる |
| オーグメンテーション | なし | `image_transforms.enable=true`（弱め） | 実データに摂動があるので保険扱い |
| steps | 3,000 | **15,000**（batch 32）| 480k サンプル ≒ 1.3 epoch。過適合を避ける |
| batch | 1（T4） | **32**（A6000 48GB） | 1 枚で余裕がある |
| LoRA r / α | 16 / 16 | **32 / 32** | データ量に合わせて容量を増やす |
| lr | 3e-4 | **1e-4 → 1e-5 decay** | 長い学習に合わせて下げる |

均等間隔で取る理由: このデータセットはタスクを round-robin で書き出して
おり（`episode 0=task0, 1=task1, …`）、摂動条件も `episode_index` 方向に
並んでいる可能性が高い。連続した塊で取ると 1 条件に偏る。

#### ダウンロード量の注意

`--dataset.episodes` は**使うエピソードを絞るだけ**で、必要な動画ファイルは
各エピソードの `videos/observation.images.*/file_index` で決まる。
round-robin 書き出しなので均等間隔で選ぶとほぼ全ての動画ファイルに触る。
**本数を絞ってもダウンロード量は減らない。** 学習前に実サイズと
ディスク残量を確認すること。

```bash
CONDA=/opt/home/ohara/miniforge3/envs/parc-policy
$CONDA/bin/python - <<'PY'
from collections import defaultdict
from huggingface_hub import HfApi
info = HfApi().dataset_info(
    "lerobot/libero_plus",
    revision="f3f49f426d75030177b18778374005bc12ccd588",
    files_metadata=True,
)
by_kind = defaultdict(int)
for f in info.siblings:
    size = f.size or 0
    kind = "videos" if f.rfilename.startswith("videos/") else \
           "data" if f.rfilename.startswith("data/") else "meta/other"
    by_kind[kind] += size
total = sum(by_kind.values())
for kind, size in sorted(by_kind.items()):
    print(f"{kind:12s} {size / 2**30:8.2f} GiB")
print(f"{'合計':12s} {total / 2**30:8.2f} GiB")
PY

df -h /opt/home/ohara
```

判定ゲート: 学習後、本リポジトリのパイプライン（EGL・128×128・300 step・
公開 4 タスク × 10 ep）で **成功率 ≥ 82.5% かつ collision 悪化なし**なら採用。
公開 4 タスクは回帰ガードにしかならない（採点セットへの汎化は測れない）ことを
忘れないこと。

解像度ギャップ（学習 256×256 / PARC 128×128、`pipeline/config.py:51`）への
対策として「128 に縮小してから戻す」劣化を学習時に入れる案があるが、
lerobot の標準 transform には無くカスタム実装が要るので第 2 ラウンドに回す。

#### 学習前に wakaba で確認すること

```bash
# 1. GPU の VRAM
nvidia-smi --query-gpu=name,memory.total --format=csv

# 2. parc-policy に lerobot-train と peft があるか
/opt/home/ohara/miniforge3/envs/parc-policy/bin/python -c \
  "import peft; print('peft', peft.__version__)"
/opt/home/ohara/miniforge3/envs/parc-policy/bin/lerobot-train --help >/dev/null && echo "lerobot-train OK"

# 3. データセットの各タスクのエピソード数（10/タスク取れるか）
/opt/home/ohara/miniforge3/envs/parc-policy/bin/python - <<'PY'
from collections import Counter
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
md = LeRobotDatasetMetadata("lerobot/libero_plus",
    revision="f3f49f426d75030177b18778374005bc12ccd588")
counts = Counter()
for cell in md.episodes["tasks"]:
    name = cell if isinstance(cell, str) else str(cell[0])
    counts[name] += 1
print("タスク数:", len(counts))
for name, n in sorted(counts.items()):
    print(f"{n:4d}  {name}")
PY
```

3 は全データのダウンロードは発生しない（メタデータのみ）。
学習本体はデータセットの実体（動画）を落とすので、ディスク残量に注意する。

---

## 26. LoRA の判定ライン（ベース基準値）

追加学習の採否はこの数字に対して決める。EGL・公開 4 タスク × 10 エピソード・
300 step・seed 42・`N_ACTION_EXEC=5`（現行の既定）。所要 563 秒。

| タスク | ベース |
|---|---|
| pick_up_the_black_bowl_in_the_top_drawer_..._table_2 | 90.0% |
| pick_up_the_tomato_sauce_..._table_27 | 70.0% |
| pick_up_the_milk_..._light_15 | 90.0% |
| put_the_bowl_on_the_stove_light_11 | 80.0% |
| **overall** | **82.5%** |

### 26.1 この 82.5% を n_exec の比較に使わないこと

`n_exec=10` / OSMesa / 300 step の §23 も overall 82.5% だった。
タスクごとには 80/100/70/80 と 90/70/90/80 で ±30 pt 動いているのに
平均が一致している。10 エピソード/タスクでは `n_exec` の差は解像できない、
というだけである（§23.3 と同じ話）。

`n_exec=5` の採用根拠は 50 エピソードの stove スイープ（§25.1）であって、
この表ではない。本数が 5 倍あり、jerk が 2 回再現している方を採る。

### 26.2 評価の回し方

`PARC_WEIGHTS_DIR` で重みを差し替える。ファイルの移動は不要。

| 条件 | PARC_WEIGHTS_DIR | 出力先 |
|---|---|---|
| ベース | 未設定 | `results/base_nexec5` |
| 拡張なし | `runs/merged_lora_all40_r8` | `results/lora_all40` |
| 拡張あり | `runs/merged_lora_all40aug_r8` | `results/lora_all40aug` |

```bash
# ターミナル A（条件ごとに起動し直す）
PARC_WEIGHTS_DIR=$PWD/runs/merged_lora_all40_r8 bash tools/run_policy_server.sh

# ターミナル B
source activate_parc.sh
python -m pipeline --server-url http://127.0.0.1:8002 --track track1 \
  --n-episodes 10 --max-steps 300 --timeout 10 --seed 42 \
  --output-dir results/lora_all40 2>&1 | tee logs/eval_lora_all40.log
```

採用条件は「82.5% を上回り、collision が悪化していないこと」。
ただし公開 4 タスクは回帰ガードにすぎず、採点セットへの汎化は測れない
（§22.2）。10 エピソード/タスクの分解能も上記のとおり低いので、
差が小さい場合は本数を増やしてから判断すること。

---

## 27. LoRA 追加学習は失敗した（打ち切り）

公開 4 タスク × 10 エピソード・300 step・EGL・`N_ACTION_EXEC=5`。

| モデル | overall | drawer | tomato | milk | stove |
|---|---|---|---|---|---|
| **ベース（採用）** | **82.5%** | 90% | 70% | 90% | 80% |
| LoRA 15,000 step | 50.0% | 60% | 10% | 60% | 70% |
| LoRA 5,000 step | 45.0% | 20% | 10% | 60% | 90% |

差 32.5 pt は 40 エピソードで 3.3σ。ノイズではない。

### 27.1 潰した仮説（すべて外れ）

| 仮説 | 検証 | 結果 |
|---|---|---|
| 画像スロットが 5 -> 2 に減った | `empty_cameras=3` で 5 枚に戻して再学習 | ✗ loss が 12 点すべて元と完全一致（空スロットはマスクされ計算に寄与しない）。`updt_s` は 0.483 -> 1.099 と 2.3 倍になったので config は確かに効いていた |
| 正規化統計が再計算された | preprocessor の md5 比較 | ✗ ベースと同一。`lerobot-train` は再計算しない |
| 過学習 | step 5,000 の中間 checkpoint を評価 | ✗ 45.0% で 15,000 より悪い。損傷は序盤で起きている |
| `full_training_modules` が projection 層を作り直した | adapter の `modules_to_save` キーを数える | ✗ 0 件 |
| マージが重みを落とした（`strict=False`） | ベースとマージ後の safetensors を全テンソル比較 | ✗ 欠落 0 / 形状違い 0 / 変化 37（LoRA が触った層数と一致） |

### 27.2 残る説明と、追わない理由

学習データと推論条件のズレが残る。学習は 256×256、PARC の観測は 128×128 固定
（`pipeline/config.py:51`）で、画素数が 1/4 になる。追加学習はこの解像度差を
埋めるどころか、評価側に存在しない条件へ適合させた可能性がある。

ただし 1 仮説あたり検証に 15 分〜2 時間かかり、5 連続で外している。
提出物は追加学習なしで 82.5% に到達しており、そちらの検証時間を確保する方が
期待値が高い。**打ち切る。**

### 27.3 再開する場合に残っている道

- 解像度を揃える。学習時に 128 へ縮小してから戻す劣化を入れる
  （lerobot の標準 transform には無く、カスタム実装が要る）
- `train_expert_only=false` にして VLM 側も適応させる
- lr をさらに下げる（1e-5 台）。ただし step 5,000 で既に壊れているので
  効果は薄いと見ている

学習・マージ・評価の道具は揃っている（`tools/train_lora.sh`、
`tools/merge_lora.py`、`PARC_WEIGHTS_DIR`）ので、再開自体は容易である。

---

## 28. 3 回目の採点: 0.043 -> 0.077

`N_ACTION_EXEC` を 10 から 5 にした 1 行の変更だけで、スコアが 1.8 倍になった。

| 提出 | 変更点 | スコア | 合計時間 | /act レイテンシ |
|---|---|---|---|---|
| 1 回目 | — | 0.000（evdev のビルド失敗） | — | — |
| 2 回目 | vendoring でビルド問題を解消 | 0.043 | 192.7s | mean 0.042s / max 0.455s |
| **3 回目** | **`N_ACTION_EXEC` 10 -> 5** | **0.077** | 269.2s | mean 0.082s / max 0.450s |

エピソード構成は 2 回目と同じで、8 エピソード・うち 2 本が早期終了。
早期終了までの step は 216 / 161 -> **205 / 139** と短くなっている。

### 28.1 この結果が支持していること

`n_exec` を下げた狙いは衝突の低減だった（§25.1）。根拠は
stove 50 エピソードのスイープで collision 0.120 -> 0.080、および jerk が
独立 2 回で同じ順序を再現したことである。採点スコアがほぼ同じ方向に
1.8 倍動いたので、**衝突が主要なボトルネックだという読みは正しかった**
と考えてよい。

公開 4 タスクの成功率は 82.5% で `n_exec=10` と同値だった（§26.1）。
10 エピソード/タスクでは解像できない差が、採点セットでは効いている。
**公開 4 タスクの成功率は採点スコアの代理指標として弱い。**
以後は collision / jerk を主指標に見る。

### 28.2 次の一手の候補

`n_exec` をさらに下げるのは駄目である。スイープで `n_exec=2` は
collision 0.180 / jerk 6.899 と 3 条件で最悪だった（§25.1）。理由は
チャンク境界での再計画が増え、flow-matching が毎回ノイズを引き直すため
境界に不連続が生じることである。

つまり**問題はチャンク境界の不連続**であり、対策は「区間を短くする」
ではなく「境界を滑らかにする」方向にある。

- **temporal ensembling**（ACT の手法）: 毎 step 推論し、重なり合う
  予測を指数重みで平均する。境界で切り替える代わりに混ぜるので、
  再サンプリング由来の不連続が平均化される。jerk が測っていたのは
  まさにこの量である
- レイテンシ予算は十分ある。現状 mean 0.082s / 制限 10s

実装は `_predict_chunk()` の呼び出し頻度と `get_action()` のキュー処理を
変えるだけで、モデルには手を入れない。追加学習（§27 で失敗）より
はるかに安く、失敗のリスクも小さい。

→ 実装した。§29 を見ること。**既定はまだ off で、採点への影響は無い。**

---

## 29. temporal ensembling: jerk が 39% 下がった（4 タスク回帰待ち）

§28.2 の一手を `submission/policy_server.py` に入れ、stove 50 エピソードで
A/B を回した。**jerk は狙いどおり大きく下がった**（§29.7）。ただし採用の前に
公開 4 タスクの回帰確認が要るので、既定はまだ off のままである。

### 29.1 何をしているか

ステップ t で実行する action の候補は 1 つではない。t で推論したチャンクの
先頭、t-1 で推論したチャンクの 2 番目、t-2 の 3 番目 … はすべて
「ステップ t で取るべき action」の推定値である。従来はこのうち 1 つだけを
使い、チャンク境界で別の推論の結果へ乗り換えていた。flow-matching は推論の
たびにノイズを引き直すので、そこで不連続が出る。

temporal ensembling は乗り換える代わりに全部を指数重みで混ぜる。
重みは ACT と同じ `w_i = exp(-m * i)`（i=0 が最も古い予測）。

これは §25.1 で読み違えていた点への対処である。n_exec を下げても
良くならなかったのは open-loop 区間の長さが問題だったからではなく、
境界の不連続が問題だったからで、n_exec=2 は境界の数を増やすだけだった
（collision 0.180 / jerk 6.899 と 3 指標すべてで最悪）。

### 29.2 つまみ

| 環境変数 | 既定 | 意味 |
|---|---|---|
| `PARC_ENSEMBLE` | `0`（無効） | 1 で有効。有効時 `N_ACTION_EXEC` は使わない |
| `PARC_ENS_H` | `16` | 各チャンクから何ステップ先まで積むか |
| `PARC_ENS_QUERY` | `1` | 何ステップおきに推論するか。1 が ACT 本来の設定 |
| `PARC_ENS_M` | `0.01` | 重みの減衰。ACT の k と同符号・同値 |
| `PARC_ENS_GRIPPER` | `1` | 0 で gripper（action[6]）だけ平均せず最新を使う |

同時に重なる予測の本数は概ね `H / PARC_ENS_QUERY` である。

### 29.3 ACT からわざと変えた点

ACT は chunk 全長（100）を積むが、既定を 50 ではなく **16**（20fps で 0.8 秒）
にした。ACT の action は関節の**絶対位置**で、古い予測でも「同じ目標姿勢」の
推定値なので陳腐化しにくい。PARC の action は**差分**（dx, dy, dz, …）なので、
H ステップ前の観測に基づく予測は「今どこにいるか」の想定がずれたぶん
系統的に外れる。古い予測を混ぜすぎると velocity にバイアスが乗る。

同じ理由で `m` の符号も自明ではない。ACT の `k=+0.01` は**古い予測を重く見る**
向きである（i=0 が最古で重み 1、新しいほど小さい。50 本でも 1:0.61 なので
実質はほぼ一様平均）。差分 action では逆に新しい予測を重く見る（m < 0）ほうが
筋が通る可能性がある。既定は ACT に合わせたうえで、符号を A/B の軸に入れた。

gripper も同様である。ACT の gripper は連続的な開度だが、LIBERO の gripper は
実質 2 値（±1）なので、平均すると開閉の切り替わりが数ステップ鈍る。
既定は ACT に合わせて平均するが、`PARC_ENS_GRIPPER=0` で切れる。

### 29.4 コストは合計時間だけで、10 秒制約には効かない

`query=1` は毎ステップ推論するので、推論回数は n_exec=5 の 5 倍になる。
ただし効くのは**合計時間**だけである。制約は 1 リクエスト 10 秒で、
平均でも累積でもない（README のタイムアウト仕様）。1 推論は採点実測で
max 0.455 秒なので、1 リクエストあたりの余裕は変わらない。

| | n_exec=5（3 回目の採点） | ensembling query=1 の見込み |
|---|---|---|
| 1 リクエスト最大 | 0.450s（制限 10s） | 同じ。推論 1 回ぶんは変わらない |
| /act 平均 | 0.082s | 約 0.45s（毎回が推論になる） |
| 8 エピソード合計 | 269.2s | 1,000〜1,200s 程度 |

合計時間に上限がある証拠は無い（採点ログからは読めない）。それでも
4 倍は無視できないので、効果が同等なら `PARC_ENS_QUERY=2` を選ぶ。

### 29.5 測る手順

`tools/sweep_ensemble.sh` を追加した。§25.1 の n_exec スイープと同じ
プロトコル（stove・50 エピソード・300 step・seed 42・EGL）なので、
§25.1 の表とそのまま比較できる。

```bash
nohup bash tools/sweep_ensemble.sh > "logs/sweep_ens_$$.log" 2>&1 &
```

既定の 4 条件は次のとおり。1 条件あたり base が約 5.5 分、ensembling は
約 25 分（毎ステップ推論のため）で、合計 1 時間半ほどを見ておく。

| ラベル | 設定 | 狙い |
|---|---|---|
| `base` | つまみ無し | 同じ日・同じ機械での対照を取り直す |
| `h16` | `PARC_ENSEMBLE=1` | ACT そのまま |
| `h8` | `+ PARC_ENS_H=8` | 重ねる本数を半分に（§29.3 の陳腐化） |
| `h16new` | `+ PARC_ENS_M=-0.1` | 新しい予測を重く見る（同じ理由の別の当て方） |

**主指標は collision と jerk である。成功率ではない。** 3 回目の採点で、
公開 4 タスクの成功率は n_exec=10 と 5 で同値なのに採点スコアは 1.8 倍
動いた（§28.1）。50 エピソードでも success の標準誤差は 5.7 pt ある。

判定:

- **jerk が下がり collision も下がる** → 採用。既定を on にして 4 回目を出す
- **jerk だけ下がって collision が動かない** → 滑らかにはなったが接触の原因は
  別。ensembling では届かない
- **どれも動かない** → この軸も閉じる。§27.3 の追加学習へ戻る

条件の指定は `PARC_SWEEP_CONDS="ラベル:VAR=値,VAR=値 …"` で差し替えられる。
`PARC_ENS_QUERY` や `PARC_ENS_GRIPPER` を振るのは、上の 4 条件で
方向が見えてからでよい。

### 29.6 実装まわりで踏まないようにした穴

- **エピソード境界**: 予測バッファを持ち越すと前エピソードの action が次の
  冒頭に流れ込む。1mm 判定なのでそのまま衝突失格になる。`reset()` で
  `_clear_episode_state()` を呼んで捨てる
- **warmup**: 起動時に実推論を 2 回強制する処理は、キューではなく
  エピソード状態ごと捨てるように変えた。そうしないと `PARC_ENS_QUERY > 1`
  のとき 2 回目が既存の予測から作られ、実推論にならない
- **条件の取り違え**: 環境変数の渡し損ねは静かに起きる。サーバーは起動時に
  `[MyPolicy]   ensemble=…` を出し、`sweep_ensemble.sh` は条件ごとに
  この行を読んで要求と食い違っていたら中止する。スイープは条件ごとに
  `PARC_ENS_*` を `env -u` で消してから設定するので、シェルに残った値も
  混入しない（§23.3 で「同じサーバーに 3 条件とも当ててしまった」のと
  同種の事故を防ぐ）

`tests/test_temporal_ensemble.py` に単体テストを置いた。モデルは要らない
（`_predict_chunk()` を既知のチャンクに差し替える）ので、評価環境でも
parc-policy でも走る。**既定が off であること**、加重平均が手計算と一致する
こと、`reset()` で漏れないこと、壊れた環境変数でも起動することを固定している。

```bash
python -m pytest tests/test_temporal_ensemble.py -q
```

---

### 29.7 結果: jerk が 30〜39% 下がった

2026-08-09、yamabuki、stove・50 エピソード・300 step・seed 42・EGL。
1 条件あたり base 5.5 分 / ensembling 25 分、合計 1 時間半。

| 条件 | 設定 | success | collision | cartesian | jerk | sparc |
|---|---|---|---|---|---|---|
| base | 現行の既定（n_exec=5） | 0.80 | 0.180 | 0.821 | 5.932 | −2.362 |
| h16 | ACT そのまま | 0.88 | 0.120 | 0.804 | **4.129**（−30%）| −2.338 |
| **h8** | `PARC_ENS_H=8` | 0.84 | 0.140 | 0.836 | **3.632**（−39%）| −2.356 |
| h16new | `PARC_ENS_M=-0.1` | 0.84 | 0.140 | 0.820 | **3.651**（−38%）| −2.327 |

**読めるのは jerk だけである。** 判定の根拠は次のとおり。

- **jerk は実効果**。同一条件の再測定でのぶれは 2.6%（下記）、§25.1 の
  n_exec 差でも 6〜13% だったのに対し、ここは 30〜39% ある。桁が違う
- **collision は読めない**。0.180 → 0.120 / 0.140 / 0.140 は最大でも 6 pt で、
  下記の対照ドリフト（10 pt）より小さい。3 条件を束ねても 0.77σ にしかならない
- **success も読めない**。+8 / +4 / +4 pt はいずれも 1.1σ 未満

#### 対照が 10 pt ドリフトした

同じ設定（n_exec=5・stove・50 エピソード・seed 42・EGL）を §25.1 と今回で
2 回測って、こう動いた。

| | §25.1 | 今回の base | 差 |
|---|---|---|---|
| success | 0.92 | 0.80 | −12 pt（1.75σ） |
| collision | 0.080 | 0.180 | +10 pt（1.50σ） |
| jerk | 6.091 | 5.932 | **−2.6%** |

success と collision は別々の変化ではなく、**約 5 エピソードが「成功」から
「衝突失格」へ移った 1 つの変化**である。GPU は A6000 で §2 の記録と同じ、
他ジョブの相乗りも無く、`PARC_WEIGHTS_DIR` も未設定だった（サーバーの
`/proc/<pid>/environ` で確認）。したがってこれは測定そのもののばらつきである。

ここから 2 つ従う。

1. **判定は今回の `base` 行に対して行う。** 記録値と比べてはいけない。
   base を条件に入れておいたのはこのためで、そこは機能した
2. **§25.1 が `n_exec=5` を採った根拠のうち「collision 0.120 → 0.080」は、
   記録されているより弱い。** そのドリフトと同じ幅である。ただし §25.1 の
   決め手は jerk が独立 2 回で再現したことであり、そちらは今回も揺れていない
   （6.091 → 5.932）。3 回目の採点が 0.043 → 0.077 と動いた事実も独立した
   裏付けなので、結論は変わらない

#### 新しい予測を重く見るほうが良い

h8（3.632）と h16new（3.651）が並び、ACT そのままの h16（4.129）より 12%
低い。この 12% は対照のぶれ 2.6% より十分大きいので、順序は実物である。

h8 は古い予測を捨て、h16new は古い予測の重みを下げる。**どちらも「最近の
予測を重く見る」という同じ操作**であり、それが効いた。§29.3 で
「差分 action では古い予測が陳腐化するので m を負にするほうが筋が通る」と
書いた読みが、2 通りの当て方で支持されたことになる。ACT の重み付けを
そのまま持ってくるのは、この問題設定では最適ではない。

#### レイテンシは問題にならなかった

| | /act mean | /act max | slow(>2s) |
|---|---|---|---|
| base | 0.069s | 0.366s | 0 |
| ensembling（3 条件とも） | 0.24s | 0.30s | 0 |

毎ステップ推論でも 1 リクエストの最悪値は 0.30 秒（制限 10 秒）で、むしろ
base の 0.366 秒より小さい。GPU が連続稼働で温まるためと思われる。
**§29.4 で懸念した 10 秒制約への影響は無い。** 合計時間は 3.5 倍になる。

エピソード長は base 85〜92 step に対し ensembling 87〜95 step で、
「差分の平均で動きが縮んで遅くなる」効果は stove では出ていない。
ただし stove は 90 step で終わる短いタスクである。300 step 打ち切りが
6/8 を占める採点セットではここが効きうるので、下の回帰確認で見る。

### 29.8 採用の判断と、その前に必要な確認

**採用する方向である。** 理由は採点スコアの構成にある。2 回目・3 回目の採点は
成功 0 本で、0.043 / 0.077 はほぼすべて軌跡系の部分点である（§22.3）。
いま 39% 改善したのは、まさにその部分点を構成する指標のひとつである。
成功率が動かなくてもスコアが動きうる、という点で追加学習とは性質が違う。

重みは非公開なので効果の大きさは予測できない。それでも、
n_exec の 1 行変更（jerk −6%）が採点を 1.8 倍にした前例がある。

ただし**このスイープは stove 1 タスクしか見ていない**。既定を変えて提出する
前に、公開 4 タスクの回帰確認（§26 のゲート: 82.5%）を通す。

```bash
PARC_SWEEP_TASK=all PARC_SWEEP_PREFIX=t4 PARC_SWEEP_EPISODES=10 \
  PARC_SWEEP_CONDS="base: h8:PARC_ENSEMBLE=1,PARC_ENS_H=8 h16new:PARC_ENSEMBLE=1,PARC_ENS_M=-0.1" \
  nohup bash tools/sweep_ensemble.sh > "logs/sweep_t4_$$.log" 2>&1 &
```

3 条件で約 1 時間 20 分。ここで見るのは 2 つである。

- **他タスクを壊していないこと。** stove は 4 タスクで最も短く、
  milk / tomato は成功まで 150〜230 step かかる。動きが縮む効果があるなら
  そちらに出る
- **h8 と h16new のどちらを採るか。** stove では jerk 3.632 vs 3.651 で
  差が付かなかった。独立なタスクセットでもう一度並べれば決められる
  （jerk が 2 回同じ順序を出したら採る、という §25.1 と同じ決め方）

回帰が通ったら `TEMPORAL_ENSEMBLE` の既定を on にし、勝った側の
`ENSEMBLE_HORIZON` / `ENSEMBLE_M` を既定へ入れて 4 回目を提出する。
落ちたら stove 限定の改善として記録し、既定は変えない。
