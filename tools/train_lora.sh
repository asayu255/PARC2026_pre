#!/usr/bin/env bash
#
# SmolVLA を LIBERO-plus 全 40 タスクで LoRA 追加学習する。
#
#   bash tools/train_lora.sh --dry-run       # コマンドを表示するだけ
#   nohup bash tools/train_lora.sh > logs/train_lora.log 2>&1 &
#
# 主な環境変数（既定値は ENVIRONMENT_SETUP.md 25.2 の設計に対応）:
#   PARC_LORA_R=8                各 A/B 用。lora_alpha は CLI から設定できないので下記参照
#   PARC_LORA_STEPS=15000
#   PARC_LORA_BATCH=32
#   PARC_LORA_EP_PER_TASK=60
#   PARC_LORA_LR=1e-4
#   PARC_LORA_AUG=0              1 で image_transforms を有効化
#   PARC_LORA_TAG=all40          出力先 runs/lora_<tag> の識別子
#   CUDA_VISIBLE_DEVICES=0       もう 1 枚は評価用に空けておく
#
# lora_alpha について:
#   lerobot 0.4.4 の PEFT 設定 dataclass（lerobot/configs/default.py）は
#   target_modules / full_training_modules / method_type / init_type / r の
#   5 つしか持たず、lora_alpha を CLI からも設定ファイルからも渡せない。
#   _build_peft_config() は LoraConfig(**config_dict) を呼ぶだけなので、
#   lora_alpha は peft の既定値のままになる。LoRA の実効強度は
#   alpha/r なので、r を上げると強度が下がる。r=<peft の既定 alpha> に
#   すると強度 1.0 になる。既定の 8 はそれを狙った値である。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

R="${PARC_LORA_R:-8}"
STEPS="${PARC_LORA_STEPS:-15000}"
BATCH="${PARC_LORA_BATCH:-32}"
EP_PER_TASK="${PARC_LORA_EP_PER_TASK:-60}"
LR="${PARC_LORA_LR:-1e-4}"
FINAL_LR="${PARC_LORA_FINAL_LR:-1e-5}"
WARMUP="${PARC_LORA_WARMUP:-500}"
TAG="${PARC_LORA_TAG:-all40}"
WORKERS="${PARC_LORA_WORKERS:-8}"
SEED="${PARC_LORA_SEED:-42}"

# 動画デコードのバックエンド。
# ノートブックは torchcodec を使うが、parc-policy には FFmpeg の共有
# ライブラリが無く libtorchcodec_core*.so のロードに失敗する
# （OSError: Could not load this library: .../libtorchcodec_core4.so）。
# PyAV は wheel に FFmpeg を同梱しているので追加インストール無しで動く。
# torchcodec の方が速いので、FFmpeg を入れたなら torchcodec に戻してよい:
#     conda install -n parc-policy -c conda-forge ffmpeg
#     PARC_LORA_VIDEO_BACKEND=torchcodec bash tools/train_lora.sh
VIDEO_BACKEND="${PARC_LORA_VIDEO_BACKEND:-pyav}"

DATASET_REPO="lerobot/libero_plus"
DATASET_REVISION="f3f49f426d75030177b18778374005bc12ccd588"

OUT_DIR="${PARC_LORA_OUT:-$ROOT/runs/lora_${TAG}_r${R}}"
EPISODES_JSON="${PARC_LORA_EPISODES_JSON:-$ROOT/runs/episodes_${EP_PER_TASK}.json}"

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

# --- parc-policy の python を探す（run_policy_server.sh と同じ方針）---------
find_python() {
    if [ -n "${PARC_POLICY_PYTHON:-}" ]; then echo "$PARC_POLICY_PYTHON"; return; fi
    local base
    if command -v conda >/dev/null 2>&1; then
        base="$(conda info --base 2>/dev/null || true)"
        [ -n "$base" ] && [ -x "$base/envs/parc-policy/bin/python" ] \
            && { echo "$base/envs/parc-policy/bin/python"; return; }
    fi
    for c in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3"; do
        [ -x "$c/envs/parc-policy/bin/python" ] && { echo "$c/envs/parc-policy/bin/python"; return; }
    done
    echo ""
}

PY="$(find_python)"
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
    echo "ERROR: parc-policy の python が見つからない。" >&2
    echo "       PARC_POLICY_PYTHON=/path/to/envs/parc-policy/bin/python を指定すること。" >&2
    exit 1
fi
TRAIN_BIN="$(dirname "$PY")/lerobot-train"
if [ ! -x "$TRAIN_BIN" ]; then
    echo "ERROR: $TRAIN_BIN が無い。" >&2
    exit 1
fi

# 評価側シェルの副作用（venv / OSMesa / LIBERO-plus の PYTHONPATH）を持ち込まない。
# ただし HF_HUB_OFFLINE は設定しない。データセットの取得が必要なため。
CLEAN=(env -u VIRTUAL_ENV -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH)

if ! "${CLEAN[@]}" "$PY" -c 'import peft' 2>/dev/null; then
    echo "ERROR: $PY に peft が入っていない。" >&2
    echo "       pip install -c <freeze> 'peft>=0.18.0,<1.0.0' すること。" >&2
    exit 1
fi

# 動画バックエンドは学習開始の 1 秒後に初めて使われるため、壊れていても
# 15,000 step のプログレスバーが出てから落ちる。先に確かめて早く失敗させる。
case "$VIDEO_BACKEND" in
    pyav)
        BACKEND_CHECK='import av; av.open'
        BACKEND_FIX="pip install av" ;;
    torchcodec)
        BACKEND_CHECK='from torchcodec.decoders import VideoDecoder' ;;
    *)
        BACKEND_CHECK="pass" ;;
esac
if ! "${CLEAN[@]}" "$PY" -c "$BACKEND_CHECK" 2>/dev/null; then
    echo "ERROR: 動画バックエンド '$VIDEO_BACKEND' が使えない。" >&2
    "${CLEAN[@]}" "$PY" -c "$BACKEND_CHECK" 2>&1 | tail -5 | sed 's/^/    /' >&2
    if [ "$VIDEO_BACKEND" = "torchcodec" ]; then
        echo "       torchcodec は FFmpeg の共有ライブラリを要求する。" >&2
        echo "       PARC_LORA_VIDEO_BACKEND=pyav にするか、FFmpeg を入れること:" >&2
        echo "           conda install -n parc-policy -c conda-forge ffmpeg" >&2
    else
        echo "       ${BACKEND_FIX:-}" >&2
    fi
    exit 1
fi

BASE_MODEL="$ROOT/submission/model_weights"
BACKBONE="$ROOT/submission/smolvlm_backbone"
for d in "$BASE_MODEL" "$BACKBONE"; do
    [ -d "$d" ] || { echo "ERROR: $d が無い。" >&2; exit 1; }
done

# --- データセットの並行ダウンロードを防ぐ ------------------------------------
# LeRobotDataset は HF hub のキャッシュとは別に HF_LEROBOT_HOME 配下へ
# 落とし直す。そのため huggingface_hub で先に snapshot_download してあっても
# 初回は 15GiB のダウンロードが走る。この状態で 2 本目を起動すると同じ宛先へ
# 並行して書き込むことになる。実際に A/B を並列で始めて踏んだ。
#
# プロセスの有無では判定できない（このスクリプトは exec するので
# pgrep -f train_lora.sh に引っかからなくなる）。ダウンロード中の証拠である
# .incomplete ファイルを見る。
LEROBOT_HOME="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"
DATASET_DIR="$LEROBOT_HOME/$DATASET_REPO"
if [ "$DRY" = "0" ] && [ -d "$DATASET_DIR" ]; then
    if find "$DATASET_DIR" -name '*.incomplete' -newermt '-10 minutes' 2>/dev/null | grep -q .; then
        echo "ERROR: $DATASET_DIR に進行中のダウンロードがある。" >&2
        echo "       同じ宛先への並行ダウンロードになるため中止する。" >&2
        echo "       先行の学習がデータ取得を終えてから起動し直すこと:" >&2
        echo "           grep -E 'step|loss' logs/train_lora_*.log | tail" >&2
        echo "       どうしても同時に始めたい場合は宛先を分ける:" >&2
        echo "           HF_LEROBOT_HOME=\$HOME/.cache/huggingface/lerobot2 bash tools/train_lora.sh" >&2
        exit 1
    fi
fi

# --- 学習に使うエピソードを選ぶ ---------------------------------------------
mkdir -p "$ROOT/runs" "$ROOT/logs"
if [ ! -s "$EPISODES_JSON" ]; then
    echo "[train] エピソードを選択: $EP_PER_TASK 本/タスク -> $EPISODES_JSON"
    "${CLEAN[@]}" "$PY" "$ROOT/tools/inspect_dataset.py" \
        --select "$EP_PER_TASK" --out "$EPISODES_JSON"
fi
EPISODES="$(cat "$EPISODES_JSON")"
N_EPISODES="$(tr -cd ',' < "$EPISODES_JSON" | wc -c)"
N_EPISODES=$((N_EPISODES + 1))

# --- オーグメンテーション -----------------------------------------------------
# 実データに摂動が含まれている（ENVIRONMENT_SETUP.md 25.2）ので既定は無効。
# 学習カタログの外側の条件への保険として使う場合に 1 にする。
AUG=()
if [ "${PARC_LORA_AUG:-0}" != "0" ]; then
    AUG=(--dataset.image_transforms.enable=true)
fi

# --- 特徴量レイアウト ---------------------------------------------------------
# ベース config の画像スロットは 5 枚である。
#   camera1, camera2, camera3, empty_camera_0, empty_camera_1
# 推論時に policy_server.py が渡すのは front と wrist の 2 枚で、
# preprocessor の rename_map が front->camera1, wrist->camera2 に割り当てる。
# 残る camera3 / empty_camera_0 / empty_camera_1 は空のまま。
# つまり実質「実画像 2 枚 + 空 3 枚 = 5 スロット」で事前学習されている。
#
# ノートブックの input_features=null / output_features=null / empty_cameras=0 を
# そのまま使うと、データセットから再導出されて front, wrist の 2 枚だけになる。
# ベースの重みは 5 枚前提であり、vision encoder 凍結 + train_expert_only では
# rank 8 の LoRA でこのズレを吸収できない。実際に公開 4 タスクが
# 82.5% -> 50.0% に落ちた。
#
# かといってフラグを一切渡さないと、データセットのキー（front / wrist）が
# ベースの input_features（camera1..）と一致せず lerobot が弾く。
#   ValueError: Feature mismatch between dataset/environment and policy config.
#
# そこで再導出はさせた上で、空スロットの数でスロット総数をベースに合わせる。
# front + wrist + 空 N 枚 = 2 + N。ベースの 5 に合わせるなら N=3。
# keep を指定すると何も渡さない（ベース config のまま。上記の理由で失敗する）。
EMPTY_CAMERAS="${PARC_LORA_EMPTY_CAMERAS:-3}"
FEATURES=()
if [ "$EMPTY_CAMERAS" != "keep" ]; then
    FEATURES=(
        --policy.input_features=null
        --policy.output_features=null
        --policy.empty_cameras="$EMPTY_CAMERAS"
    )
fi

CMD=(
    "$TRAIN_BIN"
    --policy.path="$BASE_MODEL"
    --policy.vlm_model_name="$BACKBONE"
    --policy.push_to_hub=false
    --policy.repo_id=null
    "${FEATURES[@]}"
    --policy.freeze_vision_encoder=true
    --policy.train_expert_only=true
    --policy.optimizer_lr="$LR"
    --policy.scheduler_decay_lr="$FINAL_LR"
    --policy.scheduler_warmup_steps="$WARMUP"
    --policy.scheduler_decay_steps="$STEPS"
    --dataset.repo_id="$DATASET_REPO"
    --dataset.revision="$DATASET_REVISION"
    --dataset.episodes="$EPISODES"
    --dataset.use_imagenet_stats=false
    --dataset.video_backend="$VIDEO_BACKEND"
    "${AUG[@]}"
    --output_dir="$OUT_DIR"
    --job_name="lora_${TAG}_r${R}"
    --steps="$STEPS"
    --batch_size="$BATCH"
    --num_workers="$WORKERS"
    --seed="$SEED"
    --save_checkpoint=true
    --save_freq="${PARC_LORA_SAVE_FREQ:-5000}"
    --log_freq="${PARC_LORA_LOG_FREQ:-100}"
    --wandb.enable=false
    --peft.method_type=LORA
    --peft.r="$R"
)

echo "======================================================"
echo " LoRA 追加学習"
echo "   ベース      : $BASE_MODEL"
echo "   エピソード  : $N_EPISODES 本（$EP_PER_TASK / タスク）"
echo "   r           : $R   （lora_alpha は peft 既定。実効強度 = alpha/r）"
echo "   steps       : $STEPS   batch: $BATCH   lr: $LR -> $FINAL_LR"
echo "   拡張        : ${PARC_LORA_AUG:-0}   動画: $VIDEO_BACKEND"
if [ "$EMPTY_CAMERAS" = "keep" ]; then
    echo "   画像スロット: ベース config のまま（5 枚）"
else
    echo "   画像スロット: front + wrist + 空 $EMPTY_CAMERAS 枚 = $((2 + EMPTY_CAMERAS)) 枚（ベースは 5 枚）"
fi
echo "   出力        : $OUT_DIR"
echo "   GPU         : ${CUDA_VISIBLE_DEVICES:-0}"
echo "======================================================"

if [ "$DRY" = "1" ]; then
    echo
    echo "--dry-run のため実行しない。組み立てたコマンド:"
    printf '  %q\n' "${CMD[@]}" | head -40
    echo "  （--dataset.episodes は $N_EPISODES 要素のため省略表示）"
    exit 0
fi

exec "${CLEAN[@]}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    TOKENIZERS_PARALLELISM=false \
    "${CMD[@]}"
