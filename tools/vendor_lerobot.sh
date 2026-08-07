#!/usr/bin/env bash
#
# lerobot を submission/vendor/ へ同梱し、requirements.txt を
# 「wheel のある依存だけ」に作り直す。
#
#   bash tools/vendor_lerobot.sh
#
# なぜ同梱するのか:
#   lerobot の必須依存に pynput があり、Linux ではこれが evdev を引く。
#   evdev は wheel が一切公開されておらず必ずソースビルドになるが、
#   採点環境には Python.h (python3-dev) が無いためコンパイルに失敗する。
#   実際にこれで採点が 0 点になった。
#     fatal error: Python.h: No such file or directory
#   推論の import 連鎖に pynput/evdev は含まれないので、lerobot 本体だけを
#   置いて、pip には残りの依存を入れさせる。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PARC_POLICY_PYTHON:-}"
if [ -z "$PY" ]; then
    if command -v conda >/dev/null 2>&1; then
        base="$(conda info --base 2>/dev/null || true)"
        [ -n "$base" ] && PY="$base/envs/parc-policy/bin/python"
    fi
    [ -x "${PY:-}" ] || PY="$HOME/miniforge3/envs/parc-policy/bin/python"
fi
[ -x "$PY" ] || { echo "ERROR: parc-policy の python が見つからない。PARC_POLICY_PYTHON を指定すること。" >&2; exit 1; }

SRC="$("$PY" -c 'import lerobot, pathlib; print(pathlib.Path(lerobot.__file__).parent)')"
VER="$("$PY" -c 'import lerobot; print(lerobot.__version__)' 2>/dev/null || echo unknown)"
echo "[vendor] lerobot $VER : $SRC"

DEST="submission/vendor"
rm -rf "$DEST"; mkdir -p "$DEST"
cp -r "$SRC" "$DEST/lerobot"
find "$DEST" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$DEST" -name '*.pyc' -delete 2>/dev/null || true
echo "[vendor] 配置: $DEST/lerobot  ($(du -sh "$DEST" | cut -f1))"

# 推論の import 連鎖に現れる第三者パッケージ（静的解析で確定）。
# import 名 -> pip 名。バージョンは動作実績のある parc-policy に合わせて固定する。
PKGS="torch torchvision torchcodec transformers safetensors huggingface_hub \
accelerate peft diffusers datasets pyarrow pandas fsspec av imageio pillow \
draccus packaging numpy typing_extensions einops num2words termcolor \
jsonlines deepdiff"

{
    echo "# ポリシーサーバーの依存（必須、削除しないでください）"
    echo "fastapi>=0.68"
    echo "uvicorn>=0.15"
    echo "msgpack>=1.0"
    echo ""
    echo "# SmolVLA の推論に必要な依存。"
    echo "# lerobot 自体は vendor/ に同梱してあるのでここには書かない。"
    echo "# lerobot を pip で入れると pynput -> evdev のソースビルドが走り、"
    echo "# 採点環境には Python.h が無いため必ず失敗する。"
    echo "# バージョンは動作実績のある parc-policy 環境に合わせて固定。"
    for name in $PKGS; do
        v="$("$PY" - "$name" <<'PYX'
import sys
from importlib.metadata import version, PackageNotFoundError
try:
    print(version(sys.argv[1]))
except PackageNotFoundError:
    print("")
PYX
)"
        if [ -n "$v" ]; then
            # torch の +cu128 のようなローカル版指定子は落とす（PEP 440 上
            # ==2.10.0 は 2.10.0+cu128 にも一致するので問題ない）
            echo "${name}==${v%%+*}"
        else
            echo "[vendor] 警告: $name が parc-policy に無い。requirements から除外した。" >&2
        fi
    done
} > submission/requirements.txt

echo "[vendor] requirements.txt を再生成:"
sed 's/^/    /' submission/requirements.txt
echo
echo "次に必ず実行すること（採点環境と同条件の検証）:"
echo "    bash tools/verify_clean_env.sh"
