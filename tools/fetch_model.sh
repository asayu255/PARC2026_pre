#!/usr/bin/env bash
#
# HF Hub から checkpoint を取る。
#
#   bash tools/fetch_model.sh lerobot/pi05_libero_finetuned_v044 ~/parc_models/pi05_v044
#   bash tools/fetch_model.sh google/paligemma-3b-pt-224 ~/parc_models/pg_tok \
#        'tokenizer*' 'special_tokens_map.json' 'preprocessor_config.json'
#
# CLI の名前は huggingface_hub のバージョンで変わる（huggingface-cli -> hf）ので
# 名前に依存しない Python API を直接叩く。parc-policy の python を使うため、
# base 環境に huggingface_hub が入っていなくても動く。
#
# 落としたあとにファイル一覧とサイズを出す。zip 制限は 20 GB / 展開 40 GB なので、
# 重みが fp32 で来ていないか（7B なら 28 GB になる）ここで気づけるようにしている。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

REPO="${1:-}"
DEST="${2:-}"
if [ -z "$REPO" ] || [ -z "$DEST" ]; then
    echo "usage: bash tools/fetch_model.sh <hf-repo-id> <dest-dir> [allow-pattern ...]" >&2
    exit 2
fi
shift 2

# --- parc-policy の python を探す（run_policy_server.sh と同じ規則）-----------
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

mkdir -p "$DEST"
echo "[fetch] python : $PY"
echo "[fetch] repo   : $REPO"
echo "[fetch] dest   : $DEST"
[ "$#" -gt 0 ] && echo "[fetch] allow  : $*"

# 途中で切れても resume できる（snapshot_download は既存ファイルを飛ばす）。
REPO="$REPO" DEST="$DEST" ALLOW="$*" "$PY" - <<'PYEOF'
import os
import sys

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError

repo = os.environ["REPO"]
allow = os.environ["ALLOW"].split() or None
try:
    path = snapshot_download(
        repo_id=repo,
        local_dir=os.environ["DEST"],
        allow_patterns=allow,
        max_workers=8,
    )
except GatedRepoError:
    # google/paligemma-3b-pt-224 がこれ。π0 系の tokenizer は全部そこを見る。
    # 巨大なトレースバックの下に埋もれると原因が読めないので、やることだけ出す。
    print(
        f"\n[fetch] {repo} は gated repo で、アクセス許可とログインが要る。\n"
        f"[fetch]   1. https://huggingface.co/{repo} でライセンスに同意する\n"
        f"[fetch]   2. {sys.executable.rsplit('/', 1)[0]}/huggingface-cli login\n"
        f"[fetch]      （https://huggingface.co/settings/tokens の read トークン）\n"
        f"[fetch]   3. このコマンドをもう一度実行する\n",
        file=sys.stderr,
    )
    raise SystemExit(3)
print(f"[fetch] done: {path}", file=sys.stderr)
PYEOF

echo
echo "[fetch] --- 中身（1 MB 以上を大きい順）---"
find "$DEST" -type f -size +1M -printf '%10s  %p\n' 2>/dev/null | sort -rn | head -20 \
    | awk '{printf "%8.2f GB  %s\n", $1/1073741824, $2}'
echo "[fetch] --- 合計 ---"
du -sh "$DEST"
