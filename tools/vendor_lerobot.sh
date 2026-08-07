#!/usr/bin/env bash
#
# lerobot を submission/vendor/ へ同梱し、requirements.txt を
# 「wheel のある依存だけ」に作り直す。
#
# submission/vendor/ と submission/requirements.txt はこのスクリプトの
# 生成物で、git 管理外である。管理下に置くと、再生成のたびに作業ツリーが
# 汚れて git pull が
#   error: Your local changes to the following files would be overwritten
# で止まる。実際にそれで修正が反映されないまま古いスクリプトが動いた。
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

# 同梱するモジュール。いずれも純 Python で、pip では入れられない
# （または入れると採点環境で失敗する）もの。
#   lerobot   : 必須依存 pynput -> evdev が wheel 無しでソースビルドになる
#   num2words : transformers の SmolVLM プロセッサが要求する。
#               依存の docopt に wheel が無い
#   docopt    : 上記の依存。wheel が一切公開されていない
VENDOR_MODULES="${PARC_VENDOR_MODULES:-lerobot num2words docopt}"

DEST="submission/vendor"
rm -rf "$DEST"; mkdir -p "$DEST"

for mod in $VENDOR_MODULES; do
    loc="$("$PY" - "$mod" <<'PYX'
import importlib.util, pathlib, sys
spec = importlib.util.find_spec(sys.argv[1])
if spec is None or not spec.origin:
    print(""); raise SystemExit
p = pathlib.Path(spec.origin)
# パッケージなら __init__.py の親、単一モジュールならそのファイル
print(p.parent if p.name == "__init__.py" else p)
PYX
)"
    if [ -z "$loc" ] || [ ! -e "$loc" ]; then
        echo "[vendor] ERROR: $mod が $PY に見つからない。" >&2
        echo "         conda activate parc-policy して pip install しておくこと。" >&2
        exit 1
    fi
    ver="$("$PY" - "$mod" <<'PYX'
import sys
from importlib.metadata import version, PackageNotFoundError
try: print(version(sys.argv[1]))
except PackageNotFoundError: print("?")
PYX
)"
    cp -r "$loc" "$DEST/"

    # dist-info も持っていく。transformers の is_xxx_available() は
    # importlib.metadata.version() で有無を判定することがあり、
    # メタデータが無いと同梱していても「未インストール」と見なされる。
    # importlib.metadata は sys.path 上の *.dist-info を探すので、
    # vendor/ に置けば認識される。
    di="$("$PY" - "$mod" <<'PYX'
import sys
from importlib.metadata import distribution, PackageNotFoundError
try:
    d = distribution(sys.argv[1])
    p = getattr(d, "_path", None)
    print(p if p else "")
except PackageNotFoundError:
    print("")
PYX
)"
    if [ -n "$di" ] && [ -d "$di" ]; then
        cp -r "$di" "$DEST/"
        echo "[vendor] $mod $ver <- $loc  (+ $(basename "$di"))"
    else
        echo "[vendor] $mod $ver <- $loc  (dist-info 無し)"
    fi
done

find "$DEST" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$DEST" -name '*.pyc' -delete 2>/dev/null || true
echo "[vendor] 配置: $DEST ($(du -sh "$DEST" | cut -f1))"

# 推論の import 連鎖に現れる第三者パッケージ（静的解析で確定）。
# import 名 -> pip 名。バージョンは動作実績のある parc-policy に合わせて固定する。
# この一覧は lerobot の import 連鎖を静的解析して作った。entry は
# policy_server.py が実際に import するものと一致させること:
#   lerobot / lerobot.policies.factory /
#   lerobot.policies.smolvla.modeling_smolvla / lerobot.configs.policies
# （factory を entry に入れ忘れて gymnasium を落とした）。75 モジュールに到達する。
# ただし静的解析は関数内の遅延 import を追えないため、最終的な正解は
# tools/verify_clean_env.sh（まっさらな venv で実際に起動する）である。
#
# 除外したもの:
#   num2words / docopt : pip ではなく vendor/ に同梱する（上記 VENDOR_MODULES）。
#               num2words は lerobot ではなく transformers の SmolVLM
#               プロセッサが要求する。依存の docopt に wheel が無い。
#   peft      : 解析には出るがクラスメソッド内の遅延 import のみ。
#               parc-policy にも入っていないが推論は動いている（成功率 87.5%）。
PKGS="torch torchvision torchcodec transformers safetensors huggingface_hub \
accelerate diffusers datasets pyarrow pandas fsspec av imageio pillow \
draccus packaging numpy typing_extensions einops termcolor tqdm \
pyserial gymnasium jsonlines deepdiff"

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
