#!/usr/bin/env bash
#
# N_ACTION_EXEC を振って 1 タスクを評価し、結果を比較する。
#
#   bash tools/sweep_nexec.sh                 # 50 / 25 / 10 を各 10 エピソード
#   bash tools/sweep_nexec.sh --dry-run       # 実行計画だけ表示
#   PARC_SWEEP_NEXEC="50 10" bash tools/sweep_nexec.sh
#   PARC_SWEEP_EPISODES=4 bash tools/sweep_nexec.sh
#
# 2 時間近くかかるので tmux か nohup 推奨:
#   nohup bash tools/sweep_nexec.sh > logs/sweep.log 2>&1 &
#
# ポリシーサーバーは run_policy_server.sh 経由で起動する。評価側の
# PYTHONPATH / LD_LIBRARY_PATH / VIRTUAL_ENV は同スクリプトが切り離すので、
# このスクリプト自身は評価側の環境（activate_parc.sh）で動いてよい。
set -uo pipefail            # -e は付けない。1 条件が失敗しても残りを続ける
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TASK="${PARC_SWEEP_TASK:-put_the_bowl_on_the_stove_light_11}"
EPISODES="${PARC_SWEEP_EPISODES:-10}"
MAX_STEPS="${PARC_SWEEP_MAX_STEPS:-600}"
SEED="${PARC_SWEEP_SEED:-42}"
PORT="${PARC_PORT:-8002}"
NEXEC_LIST="${PARC_SWEEP_NEXEC:-50 25 10}"
HEALTH_TIMEOUT="${PARC_SWEEP_HEALTH_TIMEOUT:-180}"

echo "======================================================"
echo " N_ACTION_EXEC スイープ"
echo "   タスク      : $TASK"
echo "   条件        : $NEXEC_LIST"
echo "   エピソード  : $EPISODES / 条件   最大ステップ: $MAX_STEPS   seed: $SEED"
echo "   ポート      : $PORT"
echo "   出力        : results/nexec<N>/  ログ: logs/"
echo "======================================================"

if [ "${1:-}" = "--dry-run" ]; then
    echo "--dry-run のため実行しない。"
    exit 0
fi

mkdir -p logs

SRV=""
cleanup() {
    if [ -n "$SRV" ] && kill -0 "$SRV" 2>/dev/null; then
        echo "[sweep] サーバー ($SRV) を停止"
        kill "$SRV" 2>/dev/null
        wait "$SRV" 2>/dev/null
    fi
}
trap 'echo; echo "[sweep] 中断された"; cleanup; exit 130' INT TERM
trap cleanup EXIT

health_ok() {
    env -u LD_LIBRARY_PATH curl -fsS --max-time 3 \
        "http://127.0.0.1:$PORT/health" >/dev/null 2>&1
}

wait_port_free() {
    for _ in $(seq 1 30); do
        health_ok || return 0
        sleep 1
    done
    echo "[sweep] 警告: ポート $PORT がまだ塞がっている"
}

# 評価側の環境（venv / PYTHONPATH / LIBERO_ROOT / OSMesa）
if [ -z "${LIBERO_ROOT:-}" ]; then
    echo "[sweep] activate_parc.sh を読み込む"
    # shellcheck disable=SC1091
    source ./activate_parc.sh
fi

for N in $NEXEC_LIST; do
    OUT="results/nexec${N}"
    echo
    echo "=== N_ACTION_EXEC=$N ==============================="
    rm -rf "$OUT"; mkdir -p "$OUT"

    wait_port_free
    PARC_N_EXEC="$N" PARC_PORT="$PORT" \
        bash tools/run_policy_server.sh > "logs/server_nexec${N}.log" 2>&1 &
    SRV=$!

    echo "[sweep] サーバー起動待ち (pid=$SRV, 上限 ${HEALTH_TIMEOUT}s)"
    ready=0
    for _ in $(seq 1 "$HEALTH_TIMEOUT"); do
        if ! kill -0 "$SRV" 2>/dev/null; then
            echo "[sweep] サーバーが終了した。logs/server_nexec${N}.log の末尾:"
            tail -20 "logs/server_nexec${N}.log" | sed 's/^/    /'
            break
        fi
        if health_ok; then ready=1; break; fi
        sleep 1
    done

    if [ "$ready" != "1" ]; then
        echo "[sweep] N=$N は起動できなかったので飛ばす"
        cleanup; SRV=""
        continue
    fi
    grep -E '^\[MyPolicy\]' "logs/server_nexec${N}.log" | sed 's/^/    /'

    python -m pipeline --server-url "http://127.0.0.1:$PORT" --track track1 \
        --tasks "$TASK" --n-episodes "$EPISODES" --max-steps "$MAX_STEPS" \
        --timeout 10 --seed "$SEED" --output-dir "$OUT" \
        > "logs/eval_nexec${N}.log" 2>&1
    rc=$?
    [ "$rc" = "0" ] || echo "[sweep] 評価が rc=$rc で終了。logs/eval_nexec${N}.log を確認"

    cleanup; SRV=""
done

echo
echo "======================================================"
echo " 結果"
echo "======================================================"
python3 - "$NEXEC_LIST" <<'PY'
import glob, json, sys

print(f"{'n_exec':>7} {'success':>8} {'collision':>10} {'cartesian':>10} "
      f"{'jerk(rms)':>10} {'sparc':>8}")
for n in sys.argv[1].split():
    files = sorted(glob.glob(f"results/nexec{n}/*.json"))
    if not files:
        print(f"{n:>7} {'--- 結果なし ---':>8}")
        continue
    d = json.load(open(files[-1]))
    tracks = d.get("tracks") or []
    tasks = (tracks[0].get("tasks") if tracks else []) or []
    if not tasks:
        print(f"{n:>7}   評価が失敗（tasks が空）")
        continue
    t = tasks[0]; m = t.get("metrics", {})
    def g(k): 
        v = m.get(k)
        return f"{v:.3f}" if isinstance(v, (int, float)) else "-"
    print(f"{n:>7} {t['success_rate']:>8.2f} {g('collision_rate'):>10} "
          f"{g('cartesian_path_length'):>10} {g('rms_cartesian_jerk'):>10} {g('sparc'):>8}")

print()
print("読み方:")
print("  success が上がる            -> その条件を採用")
print("  collision だけ下がる        -> open-loop が短いほど接触は減る。方向は正しい")
print("  どれも動かない              -> この軸は効かない。追加学習へ")
print("  cartesian が大きく減る      -> 無駄な動きが減った（成功に直結するとは限らない）")
PY
