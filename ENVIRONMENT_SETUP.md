# PARC 2026 現在の環境構築メモ

最終更新: 2026-08-06  
対象ホスト: `wakaba`  
リポジトリ: `/opt/home/ohara/PARC2026_pre`  
対象ブランチ: `claude/task-overview-details-g4eu4t`

この文書は、管理者権限を使えない Ubuntu サーバー上で、PARC 2026 の Track 1 評価環境と SmolVLA ポリシー実行環境を構築した現在の状態をまとめたものである。

## 1. 現在の到達点

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

### 未完了のもの

- 成功率の改善（追加学習）

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
- `MUJOCO_GL=osmesa`
- `PYOPENGL_PLATFORM=osmesa`

主な設定値:

```bash
export PARC_OSMESA_ROOT="$HOME/.local/parc-osmesa/root"
export PARC_OSMESA_LIB="$PARC_OSMESA_ROOT/usr/lib/x86_64-linux-gnu"
export MAGICK_HOME="/opt/home/ohara/miniforge3/envs/parc-venv-bootstrap"
export LD_LIBRARY_PATH="$PARC_OSMESA_LIB:$MAGICK_HOME/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
```

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
  --max-steps 600 \
  --timeout 10 \
  --seed 42 \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_FILE"
```

`--verbose` はルートロガーを DEBUG にし、Numba の大量ログが出るため通常は付けない。

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
