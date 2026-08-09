#!/usr/bin/env bash
#
# temporal ensembling の A/B を条件ごとに回して比較する。
#
#   bash tools/sweep_ensemble.sh                     # 既定 4 条件
#   bash tools/sweep_ensemble.sh --dry-run           # 実行計画だけ表示
#   PARC_SWEEP_EPISODES=10 bash tools/sweep_ensemble.sh
#   PARC_SWEEP_CONDS="base: ens:PARC_ENSEMBLE=1" bash tools/sweep_ensemble.sh
#
# 条件の書式は "ラベル:VAR=値,VAR=値"。値に空白は使えない。
# ラベルだけ（`base:`）を書くと、つまみを一切設定しない現行の既定になる。
#
# 比較の主指標は collision と jerk である。成功率ではない。
# 3 回目の採点（§28）で、公開 4 タスクの成功率は n_exec=10 と 5 で同値なのに
# 採点スコアは 1.8 倍動いた。10 エピソード/タスクの成功率は採点スコアの
# 代理指標として弱い。一方 jerk は 1 エピソード約 90 step × 本数の平均で
# 分散が小さく、§23.3 と §25.1 の独立な 2 回で順序が再現している。
#
# 既定は §25.1 の n_exec スイープと同じプロトコル（stove・50 エピソード・
# 300 step・seed 42・EGL）である。そのまま下の記録値と比べられる。
#
#     n_exec  success  collision  cartesian  jerk(rms)  sparc
#         10     0.84      0.120      0.842      6.470   -2.421
#          5     0.92      0.080      0.838      6.091   -2.371   <- 現行の既定
#          2     0.80      0.180      0.826      6.899   -2.407
#
# ensembling は query=1 だと毎ステップ推論するので、1 条件あたりの所要時間は
# n_exec=5 の約 5 倍（stove 50 エピソードで 5.5 分 -> 25 分前後）になる。
# 1 リクエストは 0.5 秒程度で 10 秒制限には余裕があるが、合計時間は伸びる。
# nohup 推奨:
#   nohup bash tools/sweep_ensemble.sh > "logs/sweep_ens_$$.log" 2>&1 &
set -uo pipefail            # -e は付けない。1 条件が失敗しても残りを続ける
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TASK="${PARC_SWEEP_TASK:-put_the_bowl_on_the_stove_light_11}"
EPISODES="${PARC_SWEEP_EPISODES:-50}"
MAX_STEPS="${PARC_SWEEP_MAX_STEPS:-300}"
SEED="${PARC_SWEEP_SEED:-42}"
PORT="${PARC_PORT:-8002}"
HEALTH_TIMEOUT="${PARC_SWEEP_HEALTH_TIMEOUT:-180}"

# 既定の 4 条件。
#   base    現行の既定（n_exec=5）。同じ日・同じ機械での対照を取り直す
#   h16     ACT そのまま（毎ステップ推論・16 本・m=+0.01）
#   h8      重ねる本数を半分に。差分 action は古い予測ほど陳腐化するので、
#           短いほうが良い可能性がある
#   h16new  m を負にして新しい予測を重く見る。上と同じ理由の別の当て方
CONDS="${PARC_SWEEP_CONDS:-base: h16:PARC_ENSEMBLE=1 h8:PARC_ENSEMBLE=1,PARC_ENS_H=8 h16new:PARC_ENSEMBLE=1,PARC_ENS_M=-0.1}"

# 条件ごとに必ず消す。前の条件の値が残っていると、意図と違う設定で
# 評価してしまい、数字が間違っていることに気づけない。
KNOBS=(PARC_ENSEMBLE PARC_ENS_H PARC_ENS_QUERY PARC_ENS_M PARC_ENS_GRIPPER PARC_N_EXEC)
CLEAR=(); for k in "${KNOBS[@]}"; do CLEAR+=(-u "$k"); done

LABELS=""
for spec in $CONDS; do LABELS="$LABELS ${spec%%:*}"; done

echo "======================================================"
echo " temporal ensembling スイープ"
echo "   タスク      : $TASK"
echo "   条件        :"
for spec in $CONDS; do
    label="${spec%%:*}"; assign="${spec#*:}"
    printf '                 %-8s %s\n' "$label" "${assign:-（つまみ無し = 現行の既定）}"
done
echo "   エピソード  : $EPISODES / 条件   最大ステップ: $MAX_STEPS   seed: $SEED"
echo "   ポート      : $PORT"
echo "   出力        : results/ens_<ラベル>/  ログ: logs/"
echo "======================================================"

if [ "${1:-}" = "--dry-run" ]; then
    echo "--dry-run のため実行しない。"
    exit 0
fi

mkdir -p logs

# --- 多重起動の防止 ---------------------------------------------------------
# sweep_nexec.sh と同じロックを使う。別種のスイープでも、同時に走れば同じ
# ポートと同じ出力先を奪い合う。先発だけがポートを握り、後発のサーバーは
# bind に失敗して死ぬ一方で /health は先発が応答するため、全条件が同じ
# サーバーに対して評価され、全部同じ結果になる。
LOCK="$ROOT/.sweep.lock"
PIDFILE="$ROOT/.sweep.pid"
exec 9>>"$LOCK" || true          # >> にする。> だと保持者の PID を消してしまう
if command -v flock >/dev/null 2>&1; then
    if ! flock -n 9; then
        holder="$(cat "$PIDFILE" 2>/dev/null || true)"
        echo "[sweep] 既に別のスイープが動いている。中止する。"
        if [ -n "$holder" ] && kill -0 "$holder" 2>/dev/null; then
            echo "    pid=$holder  $(tr '\0' ' ' < "/proc/$holder/cmdline" 2>/dev/null | cut -c1-90)"
            echo "        止めるには: kill $holder"
        else
            echo "        止めるには: pkill -f sweep_"
        fi
        exit 1
    fi
    echo "$$" > "$PIDFILE"
fi

SRV=""
cleanup() {
    rm -f "${PIDFILE:-}" 2>/dev/null
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

# ポートを LISTEN している PID を返す。pkill -f 'policy_server.py' は使わない
# （同じ文字列を含む無関係なプロセスまで巻き込む。実際に自分のシェルを落とした）。
port_pids() {
    if command -v ss >/dev/null 2>&1; then
        ss -ltnpH "sport = :$PORT" 2>/dev/null \
            | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u
    elif command -v lsof >/dev/null 2>&1; then
        lsof -ti "tcp:$PORT" -sTCP:LISTEN 2>/dev/null | sort -u
    fi
}

require_port_free() {
    health_ok || return 0

    local pids
    pids="$(port_pids)"
    echo "[sweep] ポート $PORT が既に使われている。"
    for pid in $pids; do
        echo "    pid=$pid  $(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-100)"
    done

    if [ "${PARC_SWEEP_KILL_STALE:-0}" != "0" ] && [ -n "$pids" ]; then
        echo "[sweep] PARC_SWEEP_KILL_STALE=1 のため、この PID だけを停止する"
        # shellcheck disable=SC2086
        kill $pids 2>/dev/null
        for _ in $(seq 1 30); do
            health_ok || { echo "[sweep] 解放された"; return 0; }
            sleep 1
        done
        # shellcheck disable=SC2086
        kill -9 $pids 2>/dev/null
        for _ in $(seq 1 10); do
            health_ok || { echo "[sweep] 解放された (SIGKILL)"; return 0; }
            sleep 1
        done
    fi

    echo
    echo "[sweep] 中止する。古いサーバーが応答していると、新しい設定ではなく"
    echo "        そちらに対して評価が走り、誤った結果になる。"
    echo "        先に停止すること:"
    [ -n "$pids" ] && echo "            kill $pids"
    echo "        または自動で停止させる:"
    echo "            PARC_SWEEP_KILL_STALE=1 bash tools/sweep_ensemble.sh"
    return 1
}

# 評価側の環境（venv / PYTHONPATH / LIBERO_ROOT / EGL）
if [ -z "${LIBERO_ROOT:-}" ]; then
    echo "[sweep] activate_parc.sh を読み込む"
    # shellcheck disable=SC1091
    source ./activate_parc.sh
fi

require_port_free || exit 1

for spec in $CONDS; do
    label="${spec%%:*}"
    assign="${spec#*:}"
    [ "$assign" = "$spec" ] && assign=""          # ':' が無い書き方も許す
    OUT="results/ens_${label}"
    SRVLOG="logs/server_ens_${label}.log"
    echo
    echo "=== $label : ${assign:-（つまみ無し）} ==============================="
    rm -rf "$OUT"; mkdir -p "$OUT"

    require_port_free || exit 1
    # shellcheck disable=SC2086
    env "${CLEAR[@]}" ${assign//,/ } PARC_PORT="$PORT" \
        bash tools/run_policy_server.sh > "$SRVLOG" 2>&1 &
    SRV=$!

    echo "[sweep] サーバー起動待ち (pid=$SRV, 上限 ${HEALTH_TIMEOUT}s)"
    ready=0
    for _ in $(seq 1 "$HEALTH_TIMEOUT"); do
        if ! kill -0 "$SRV" 2>/dev/null; then
            echo "[sweep] サーバーが終了した。$SRVLOG の末尾:"
            tail -20 "$SRVLOG" | sed 's/^/    /'
            break
        fi
        if health_ok; then ready=1; break; fi
        sleep 1
    done

    if [ "$ready" != "1" ]; then
        echo "[sweep] $label は起動できなかったので飛ばす"
        cleanup; SRV=""
        continue
    fi

    # /health に応答しているのが自分の起動したサーバーか確認する。
    # run_policy_server.sh は exec するので $SRV がそのまま python の PID。
    owner="$(port_pids)"
    if [ -n "$owner" ] && ! printf '%s\n' $owner | grep -qx "$SRV"; then
        echo "[sweep] ポート $PORT に応答しているのは自分が起動したサーバーではない。"
        echo "        起動したのは pid=$SRV、実際の占有は pid=$owner"
        echo "        誤ったサーバーを評価してしまうため中止する。"
        cleanup; SRV=""
        exit 1
    fi
    grep -E '^\[MyPolicy\]' "$SRVLOG" | sed 's/^/    /'

    # サーバーが実際にその設定で立ったかを起動ログで確かめる。環境変数の
    # 渡し損ねは静かに起きて、条件が違うことに気づけないまま数字が出る。
    ens_line="$(grep -o 'ensemble=.*' "$SRVLOG" | head -1)"
    case "$assign" in
        *PARC_ENSEMBLE=1*)
            if [ "${ens_line#ensemble=off}" != "$ens_line" ]; then
                echo "[sweep] $label は ensembling を要求したのにサーバーは off で起動した。"
                echo "        起動ログ: $ens_line"
                cleanup; SRV=""; exit 1
            fi ;;
        *)
            if [ "${ens_line#ensemble=on}" != "$ens_line" ]; then
                echo "[sweep] $label はつまみ無しのはずなのに ensembling が有効になっている。"
                echo "        起動ログ: $ens_line  （シェルに PARC_ENSEMBLE が残っていないか）"
                cleanup; SRV=""; exit 1
            fi ;;
    esac

    python -m pipeline --server-url "http://127.0.0.1:$PORT" --track track1 \
        --tasks "$TASK" --n-episodes "$EPISODES" --max-steps "$MAX_STEPS" \
        --timeout 10 --seed "$SEED" --output-dir "$OUT" \
        > "logs/eval_ens_${label}.log" 2>&1
    rc=$?
    [ "$rc" = "0" ] || echo "[sweep] 評価が rc=$rc で終了。logs/eval_ens_${label}.log を確認"

    # レイテンシは必ず見る。1 リクエストでも 10 秒を超えるとトラックが 0 点になる。
    grep 'レイテンシ' "$SRVLOG" | tail -3 | sed 's/^/    /'

    cleanup; SRV=""
    for _ in $(seq 1 30); do health_ok || break; sleep 1; done
done

echo
echo "======================================================"
echo " 結果"
echo "======================================================"
python3 - "$LABELS" "$EPISODES" <<'PY'
import glob, json, math, sys

labels = sys.argv[1].split()
episodes = int(sys.argv[2])

print(f"{'条件':>8} {'success':>8} {'collision':>10} {'cartesian':>10} "
      f"{'jerk(rms)':>10} {'sparc':>8}")
rows = {}
for label in labels:
    files = sorted(glob.glob(f"results/ens_{label}/*.json"))
    if not files:
        print(f"{label:>8} {'--- 結果なし ---':>8}")
        continue
    d = json.load(open(files[-1]))
    tracks = d.get("tracks") or []
    tasks = (tracks[0].get("tasks") if tracks else []) or []
    if not tasks:
        print(f"{label:>8}   評価が失敗（tasks が空）")
        continue
    t = tasks[0]; m = t.get("metrics", {})
    rows[label] = (t["success_rate"], m)

    def g(k):
        v = m.get(k)
        return f"{v:.3f}" if isinstance(v, (int, float)) else "-"
    print(f"{label:>8} {t['success_rate']:>8.2f} {g('collision_rate'):>10} "
          f"{g('cartesian_path_length'):>10} {g('rms_cartesian_jerk'):>10} {g('sparc'):>8}")

if "base" in rows and len(rows) > 1:
    base_sr, base_m = rows["base"]
    print()
    print("base との差:")
    for label, (sr, m) in rows.items():
        if label == "base":
            continue
        parts = [f"success {sr - base_sr:+.3f}"]
        for key, name in (("collision_rate", "collision"), ("rms_cartesian_jerk", "jerk")):
            a, b = m.get(key), base_m.get(key)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                parts.append(f"{name} {a - b:+.3f}")
        print(f"  {label:>8}  " + "  ".join(parts))

se = math.sqrt(0.2 * 0.8 / episodes) if episodes else float("nan")
print()
print("読み方:")
print(f"  success の標準誤差は p=0.8・{episodes} エピソードで約 {se*100:.1f} pt。")
print("  この幅に収まる差は読まないこと。単独の success で採否を決めない。")
print()
print("  jerk が下がる      -> 狙いどおり。チャンク境界の不連続が減っている")
print("  collision が下がる -> 採点スコアに直接効く（採点で到達した 2 本は")
print("                        どちらも 1mm ルールで落ちている。§22.3.1）")
print("  jerk だけ下がって collision が動かない -> 滑らかにはなったが")
print("                        接触の原因は別。ensembling では届かない")
print("  どれも動かない     -> この軸も閉じる。§27.3 の追加学習へ戻る")
print()
print("採用するなら submission/policy_server.py の TEMPORAL_ENSEMBLE の既定と、")
print("ENVIRONMENT_SETUP.md §29 の表を書き換えること。")
PY
