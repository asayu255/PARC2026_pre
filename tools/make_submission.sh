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

SRC="submission"
OUT="submission.zip"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

test -f "$SRC/policy_server.py"  || { echo "ERROR: $SRC/policy_server.py が無い" >&2; exit 1; }
test -f "$SRC/requirements.txt"  || { echo "ERROR: $SRC/requirements.txt が無い" >&2; exit 1; }

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
(cd "$SRC" && zip -q -r "../$OUT" .)
echo "$OUT: $(du -h "$OUT" | cut -f1)"
echo
echo "== zip 直下の確認 =="
unzip -l "$OUT" | head -15
echo
echo "次のコマンドで検証すること:"
echo "  python validate_submission.py $OUT"
