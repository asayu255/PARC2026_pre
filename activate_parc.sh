#!/usr/bin/env bash
#
# 評価用シェルの環境設定。リポジトリ直下で `source activate_parc.sh` する。
#
#   cd ~/PARC2026_pre
#   source activate_parc.sh
#
# ポリシーサーバー側（conda の parc-policy）とは別のシェルで使うこと。
# 同じシェルで両方を有効化すると、表示上は parc-policy でも実際には
# venv 側の python が使われる場合がある（ENVIRONMENT_SETUP.md §3 参照）。

# このスクリプトの位置からリポジトリルートを決める（cd 済みでなくても動く）
PARC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PARC_ROOT

# --- 評価用 venv / PYTHONPATH / LIBERO_ROOT ---------------------------------
# env.sh は setup.sh が生成する。git 管理外なので、無い場合は同等の設定を行う。
if [ -f "$PARC_ROOT/env.sh" ]; then
    # shellcheck disable=SC1091
    source "$PARC_ROOT/env.sh"
else
    if [ -f "$PARC_ROOT/venv/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$PARC_ROOT/venv/bin/activate"
    else
        echo "[activate_parc] 警告: $PARC_ROOT/venv が無い。先に setup.sh を実行すること。" >&2
    fi
    export PYTHONPATH="$PARC_ROOT/LIBERO-plus:$PARC_ROOT:$PARC_ROOT/compe"
    export LIBERO_ROOT="$PARC_ROOT/LIBERO-plus"
fi

# --- ユーザー領域に展開した OSMesa ------------------------------------------
export PARC_OSMESA_ROOT="${PARC_OSMESA_ROOT:-$HOME/.local/parc-osmesa/root}"
export PARC_OSMESA_LIB="$PARC_OSMESA_ROOT/usr/lib/x86_64-linux-gnu"

# --- Conda 環境に入れた ImageMagick -----------------------------------------
# 別ホストで使う場合は、source する前に MAGICK_HOME を設定して上書きする。
export MAGICK_HOME="${MAGICK_HOME:-/opt/home/ohara/miniforge3/envs/parc-venv-bootstrap}"

# --- レンダラ ----------------------------------------------------------------
# 既定は egl。採点環境が EGL / NVIDIA であり（ENVIRONMENT_SETUP.md §22.4）、
# かつ OSMesa のソフトウェアレンダリングより 1 エピソードあたり 7.7 倍速い
# （83.7 秒 -> 10.9 秒、§24）。成功率は両者で区別できない範囲に収まっている。
#
# OSMesa に戻すには source する前に PARC_RENDERER=osmesa を指定する。
#   PARC_RENDERER=osmesa source activate_parc.sh
export PARC_RENDERER="${PARC_RENDERER:-egl}"
export MUJOCO_GL="$PARC_RENDERER"
export PYOPENGL_PLATFORM="$PARC_RENDERER"

# --- ユーザー領域の共有ライブラリを優先 --------------------------------------
# OSMesa の libGL は EGL のそれと衝突するので、osmesa のときだけ前に置く。
if [ "$PARC_RENDERER" = "osmesa" ]; then
    export LD_LIBRARY_PATH="$PARC_OSMESA_LIB:$MAGICK_HOME/lib:${LD_LIBRARY_PATH:-}"
else
    export LD_LIBRARY_PATH="$MAGICK_HOME/lib:${LD_LIBRARY_PATH:-}"
fi
