#!/usr/bin/env bash

# setup.sh が生成したPython環境
source "$HOME/PARC2026_pre/env.sh"

# ユーザー領域に展開したOSMesa
export PARC_OSMESA_ROOT="$HOME/.local/parc-osmesa/root"
export PARC_OSMESA_LIB="$PARC_OSMESA_ROOT/usr/lib/x86_64-linux-gnu"

# Conda環境に入れたImageMagick
export MAGICK_HOME="/opt/home/ohara/miniforge3/envs/parc-venv-bootstrap"

# ユーザー領域の共有ライブラリを優先
export LD_LIBRARY_PATH="$PARC_OSMESA_LIB:$MAGICK_HOME/lib:${LD_LIBRARY_PATH:-}"

# 配布Docker環境と同じOSMesaを使用
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
