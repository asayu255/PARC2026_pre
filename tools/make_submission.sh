#!/usr/bin/env bash
#
# 提出 zip を作る。不要物の混入と zip 構造の誤りを防ぐ。
#
#   bash tools/make_submission.sh                 # submission/ から submission.zip
#   bash tools/make_submission.sh --dry-run       # 中身を確認するだけ
#
# 防いでいる失敗:
#   - zip 直下が policy_server.py でない（ENVIRONMENT_SETUP.md 18 の「誤り」）
#   - __pycache__ / .cache / onnx 等が混入してサイズが膨らむ
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 既定は SmolVLA 版。OpenVLA-OFT 版は tools/stage_oft_submission.sh が
# submission_oft/ を作るので、そちらを指して同じ手順で zip にする。
SRC="${PARC_SUBMISSION_SRC:-submission}"
OUT="${PARC_SUBMISSION_ZIP:-submission.zip}"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

test -f "$SRC/policy_server.py"  || { echo "ERROR: $SRC/policy_server.py が無い" >&2; exit 1; }
if [ ! -f "$SRC/requirements.txt" ]; then
    echo "ERROR: $SRC/requirements.txt が無い。" >&2
    echo "       SmolVLA 版: bash tools/vendor_lerobot.sh" >&2
    echo "       OFT 版:     bash tools/stage_oft_submission.sh" >&2
    exit 1
fi
# vendor/ は lerobot を同梱するための生成物で、SmolVLA 版にしか無い。
# OFT は lerobot を使わず、依存はすべて wheel があるので pip に任せる。
if [ ! -d "$SRC/vendor" ] && [ ! -d "$SRC/vendor_oft" ]; then
    echo "ERROR: $SRC/vendor も $SRC/vendor_oft も無い。" >&2
    echo "       先に実行すること: bash tools/vendor_lerobot.sh" >&2
    exit 1
fi

echo "== 不要物を削除 =="
# 実行時に不要で、サイズだけ食うもの
find "$SRC" -name '__pycache__'   -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$SRC" -name '.cache'        -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$SRC" -name '.git'          -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$SRC" -name '.DS_Store'     -type f -delete 2>/dev/null || true
find "$SRC" -name '*.pyc'         -type f -delete 2>/dev/null || true
# transformers は safetensors から読む。ONNX / GGUF / TF / Flax 版は使わない。
find "$SRC" -type d -name 'onnx'  -prune -exec rm -rf {} + 2>/dev/null || true
find "$SRC" -type f \( -name '*.gguf' -o -name '*.onnx' -o -name '*.onnx_data' \
                    -o -name 'tf_model.h5' -o -name 'flax_model.msgpack' \
                    -o -name 'pytorch_model.bin' \) -delete 2>/dev/null || true
# checkpoint に同梱される評価成果物（推論には不要）
find "$SRC" -type d \( -name 'eval' -o -name 'runs' \) -prune -exec rm -rf {} + 2>/dev/null || true
find "$SRC" -type f \( -name '*.mp4' -o -name '*.webm' -o -name '*.gif' \) -delete 2>/dev/null || true

echo "== 構成 =="
du -sh "$SRC"
du -sh "$SRC"/* 2>/dev/null | sort -rh
echo
echo "== zip 直下に入るもの（policy_server.py が直下にあること）=="
(cd "$SRC" && ls -1)

if [ "$DRY" = "1" ]; then
    echo
    echo "--dry-run のため zip は作成しない。"
    exit 0
fi

echo
echo "== zip を作成 =="
rm -f "$OUT"
# $SRC の「中身」を zip 直下へ入れる（submission/ という階層を作らない）
#
# 圧縮率は PARC_ZIP_LEVEL で変えられる。safetensors は既に密で deflate が
# ほとんど効かないため、15 GB の OFT 版では -1（最速）にすると数十分が
# 数分になる。上限は 20 GB あり、削るべきは容量ではなく時間である。
(cd "$SRC" && zip -q -r "${PARC_ZIP_LEVEL:--6}" "../$OUT" .)
echo "$OUT: $(du -h "$OUT" | cut -f1)"
echo
echo "== zip 直下の確認 =="
# unzip -l | head だと head が閉じた時点で unzip が SIGPIPE で死に、
# set -o pipefail がそれを拾ってスクリプトごと終了する。
# 実際にそれで後続の verify_clean_env.sh が実行されなかった。
# 一度変数に受けてから切り詰める。
listing="$(unzip -l "$OUT")"
printf '%s\n' "$listing" | head -20

# validate_submission.py は sys.executable でサーバーを起動する (:554)。
# つまり validate を動かす python に lerobot が入っていないと必ず失敗する。
PY_HINT="python"
if [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    PY_HINT="$CONDA_PREFIX/bin/python"
fi

echo
if [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "警告: VIRTUAL_ENV=$VIRTUAL_ENV が有効になっている。"
    echo "      評価用 venv と parc-policy が重なっていると、表示に関わらず"
    echo "      venv 側の python が使われる（ENVIRONMENT_SETUP.md 3）。"
    echo "      検証は下のフルパスで実行すること。"
fi
echo "次のコマンドで検証すること:"
echo "  $PY_HINT validate_submission.py $OUT"
echo "  ※ validate はサーバーを sys.executable で起動する"
echo "    (validate_submission.py:554)。lerobot が入った python を使うこと。"
