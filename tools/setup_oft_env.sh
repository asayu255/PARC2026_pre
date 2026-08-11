#!/usr/bin/env bash
#
# OpenVLA-OFT+ 用の conda 環境 parc-oft を作る。
#
#   bash tools/setup_oft_env.sh
#
# なぜ環境を分けるのか:
#   OFT の checkpoint は transformers==4.40.1 を前提に保存されている
#   （config.json の transformers_version）。一方 parc-policy は lerobot 0.4.4 の
#   SmolVLA 用に transformers>=4.57.1 が入っている。両立しない。
#   OFT は lerobot を一切使わないので、環境ごと分けるのがいちばん安全である。
#
# なぜ clone するのか:
#   torch / CUDA は parc-policy のものが実機で動く実績がある。ゼロから作ると
#   その組み合わせを引き直すことになるので、clone して OFT が指定する
#   パッケージだけ差し替える。
set -euo pipefail

SRC_ENV="${PARC_OFT_SRC_ENV:-parc-policy}"
DST_ENV="${PARC_OFT_ENV:-parc-oft}"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda が見つからない。" >&2
    exit 1
fi
BASE="$(conda info --base)"
DST_PY="$BASE/envs/$DST_ENV/bin/python"

if [ -x "$DST_PY" ]; then
    echo "[oft-env] $DST_ENV は既にある。パッケージの確認だけ行う。"
else
    echo "[oft-env] $SRC_ENV を clone して $DST_ENV を作る（数分かかる）"
    conda create -y -n "$DST_ENV" --clone "$SRC_ENV"
fi

echo "[oft-env] OFT が指定する版へ差し替える"
# transformers 4.40.1 は tokenizers>=0.19,<0.20 を引く。timm は dinosiglip の
# vision backbone に必要（parc-policy には入っていない）。accelerate は
# low_cpu_mem_usage=True の shard ロードに要る。
"$DST_PY" -m pip install --no-input \
    "transformers==4.40.1" \
    "timm==0.9.10" \
    "accelerate" \
    "pillow"

echo
echo "[oft-env] --- 版の確認 ---"
"$DST_PY" - <<'PYEOF'
import importlib

for name in ("torch", "transformers", "tokenizers", "timm", "numpy", "huggingface_hub", "accelerate"):
    try:
        m = importlib.import_module(name)
        print(f"  {name:18s} {getattr(m, '__version__', '?')}")
    except Exception as exc:
        print(f"  {name:18s} 読めない: {exc}")

import torch

print(f"  cuda available     {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device             {torch.cuda.get_device_name(0)}")
PYEOF

echo
echo "[oft-env] 次: PARC_OFT_WEIGHTS=~/parc_models/oft_libero_plus bash tools/oft_smoke.sh"
