#!/usr/bin/env bash
#
# ポリシーサーバーを parc-policy の python で起動する。
# 現在のシェルの状態（venv の有無、activate_parc.sh の副作用）に依存しない。
#
#   bash tools/run_policy_server.sh                       # 既定 127.0.0.1:8002
#   PARC_N_EXEC=25 bash tools/run_policy_server.sh        # A/B
#   bash tools/run_policy_server.sh --port 8003
#   PARC_OFFLINE_TEST=1 bash tools/run_policy_server.sh   # HF キャッシュを隠す
#
# 解決している問題:
#   - source activate_parc.sh したシェルでは (venv) が残り、表示に関わらず
#     venv 側の python が使われる（ENVIRONMENT_SETUP.md 3）
#   - 同スクリプトが PYTHONPATH に LIBERO-plus を、LD_LIBRARY_PATH に
#     OSMesa と ImageMagick を入れる。これはポリシー側には不要で、
#     conda 環境のライブラリと競合しうるため引き継がない
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/submission"

# --- parc-policy の python を探す -------------------------------------------
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

# --- 評価側シェルの副作用を持ち込まない -------------------------------------
CLEAN=(env -u VIRTUAL_ENV -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH)

if ! "${CLEAN[@]}" "$PY" -c 'import lerobot' 2>/dev/null; then
    echo "ERROR: $PY に lerobot が入っていない。" >&2
    echo "       conda activate parc-policy して pip install 'lerobot[smolvla]' 済みか確認すること。" >&2
    exit 1
fi

# --- 採点環境と同条件にする（オフライン検証用）-------------------------------
OFFLINE=()
if [ "${PARC_OFFLINE_TEST:-0}" != "0" ]; then
    OFFLINE=(HF_HOME=/tmp/parc_empty_hf_home)
    echo "[run] PARC_OFFLINE_TEST: HF キャッシュを隠して起動する"
fi

echo "[run] python : $PY"
echo "[run] cwd    : $PWD"
echo "[run] flip180=${PARC_FLIP180:-1(既定)}  n_exec=${PARC_N_EXEC:-50(既定)}  debug=${PARC_DEBUG_DIR:-なし}"
echo

exec "${CLEAN[@]}" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
    TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
    TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    "${OFFLINE[@]}" \
    "$PY" policy_server.py --host "${PARC_HOST:-127.0.0.1}" --port "${PARC_PORT:-8002}" "$@"
