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
# 起動待ちの上限。既定は長めに取る。
#
# 180 秒だと足りないことがある。15 GB の checkpoint を NFS から読む最中に
# 提出物の zip（同じ 15 GB を読んで 12 GB を書く）が走ると、通常 10 秒の
# ロードが 180 秒を超え、条件が丸ごと飛んだ。飛んだことはログに出るが、
# そのログを二重起動で潰していたため気づくのが遅れた。
#
# 長くして損をするのは「本当に起動しない」ときだけで、そのときは
# サーバーのプロセスが死んで即座に検出される。
HEALTH_TIMEOUT="${PARC_SWEEP_HEALTH_TIMEOUT:-600}"

# 出力先の接頭辞。別のラウンドを回すときに前のラウンドを潰さないためのもの
# （条件ディレクトリは毎回 rm -rf される）。results/<接頭辞>_<ラベル>/ になる。
PREFIX="${PARC_SWEEP_PREFIX:-ens}"

# PARC_SWEEP_TASK=all で公開 4 タスク全部（--tasks を付けない）。
# 単一タスクは軸を切るため、4 タスクは回帰ガードのために使う（§26）。
TASK_ARGS=(--tasks "$TASK")
if [ "$TASK" = "all" ]; then
    TASK_ARGS=()
    TASK="公開 4 タスク全部"
fi

# 既定の 4 条件。
#   base    現行の既定（n_exec=5）。同じ日・同じ機械での対照を取り直す
#   h16     ACT そのまま（毎ステップ推論・16 本・m=+0.01）
#   h8      重ねる本数を半分に。差分 action は古い予測ほど陳腐化するので、
#           短いほうが良い可能性がある
#   h16new  m を負にして新しい予測を重く見る。上と同じ理由の別の当て方
CONDS="${PARC_SWEEP_CONDS:-base: h16:PARC_ENSEMBLE=1 h8:PARC_ENSEMBLE=1,PARC_ENS_H=8 h16new:PARC_ENSEMBLE=1,PARC_ENS_M=-0.1}"

# 全条件に共通で掛ける設定。書式は条件と同じ "KEY=VAL,KEY=VAL"。
#
# ラウンド全体の設定（どの重みを使うか等）を条件ごとに書き写さないためのもの。
# 下の CLEAR が PARC_WEIGHTS_DIR を含む全つまみを毎回消すので、シェルで
# export しても条件には届かない。条件文字列に毎回足すのは書き忘れが起きる:
#
#   OFT のラウンドで PARC_WEIGHTS_DIR を落とすと submission/model_weights
#   （SmolVLA）が読まれる。parc-oft の python なら import で落ちて気づけるが、
#   parc-policy なら**黙って SmolVLA を評価して、それらしい数字を出す**。
#
#   PARC_SWEEP_COMMON=PARC_WEIGHTS_DIR=$HOME/parc_models/oft_libero_plus
#
# 条件側の指定が優先される（env は後勝ちなので、共通 -> 条件 の順に並べる）。
COMMON="${PARC_SWEEP_COMMON:-}"

# 条件ごとに必ず消す。前の条件の値が残っていると、意図と違う設定で
# 評価してしまい、数字が間違っていることに気づけない。
#
# PARC_WEIGHTS_DIR も消す。これがシェルに残っていると、base を含む全条件が
# LoRA のマージ済み重みで走る。追加学習の評価（§26.2）のあと同じシェルから
# スイープを始めると起きうる事故で、しかも静かに起きる。重みを振りたい場合は
# 条件側に `label:PARC_WEIGHTS_DIR=...` と書けばよい。
KNOBS=(PARC_ENSEMBLE PARC_ENS_H PARC_ENS_QUERY PARC_ENS_M PARC_ENS_GRIPPER
       PARC_N_EXEC PARC_WEIGHTS_DIR PARC_NUM_STEPS
       PARC_OFT_GRIPPER PARC_OFT_CENTER_CROP PARC_OFT_UNNORM PARC_OFT_DEVICE_MAP
       PARC_OFT_TTA PARC_OFT_TTA_SCALES PARC_ACT_SCALE PARC_ACT_SCALE_ROT
       PARC_ACT_SLEW PARC_ACT_SLEW_ROT
       PARC_GRIP_RETRY PARC_GRIP_RETRY_AFTER PARC_GRIP_RETRY_OPEN
       PARC_GRIP_RETRY_MAX PARC_GRIP_RETRY_MIN_STEP PARC_GRIP_ROTATE
       PARC_STALL_EPS PARC_STALL_WINDOW PARC_STALL_MIN_STEP
       PARC_STALL_MAX PARC_STALL_FLUSH
       PARC_BACKBONE_DIR PARC_PI_SKIP_TF_CHECK)
# PARC_POLICY_PYTHON は意図的に消さない。どの環境の python でサーバーを
# 立てるかはラウンド全体の設定であって、条件ごとに振るものではない
# （OFT は transformers 4.40.1 の parc-oft、SmolVLA は parc-policy）。
CLEAR=(); for k in "${KNOBS[@]}"; do CLEAR+=(-u "$k"); done

LABELS=""
for spec in $CONDS; do LABELS="$LABELS ${spec%%:*}"; done

# 条件の説明。共通部があるときの「つまみ無し」は嘘になる（共通部は効いている）。
cond_desc() {
    if [ -n "$1" ]; then echo "$1"
    elif [ -n "$COMMON" ]; then echo "（共通部のみ）"
    else echo "（つまみ無し = 現行の既定）"; fi
}

echo "======================================================"
echo " temporal ensembling スイープ"
echo "   タスク      : $TASK"
echo "   条件        :"
for spec in $CONDS; do
    label="${spec%%:*}"; assign="${spec#*:}"
    printf '                 %-8s %s\n' "$label" "$(cond_desc "${spec#*:}")"
done
[ -n "$COMMON" ] && echo "   全条件共通  : $COMMON"
echo "   エピソード  : $EPISODES / 条件   最大ステップ: $MAX_STEPS   seed: $SEED"
echo "   ポート      : $PORT"
echo "   出力        : results/${PREFIX}_<ラベル>/  ログ: logs/"
echo "======================================================"

if [ "${1:-}" = "--dry-run" ]; then
    echo "--dry-run のため実行しない。"
    exit 0
fi

mkdir -p logs

# --- 前のラウンドの残骸を検出する -------------------------------------------
# 結果表は results/<接頭辞>_* を**全部**拾う。条件ディレクトリは実行時に
# rm -rf されるが、それは今回のラベルだけである。同じ接頭辞で条件を入れ替えて
# 回し直すと、前のラウンドのラベルが消えずに残り、表に混ざる。
#
# 実際に起きかけた: oft5 を TTA4 土台で回し、途中で TTA2 土台に組み直した。
# ラベルが tta4 -> tta2/slew12/slew08 と変わるので results/oft5_tta4/ が残り、
# 土台の違う行が同じ表に並ぶところだった。数字は出るので気づけない。
STALE=""
for d in "results/${PREFIX}_"*/; do
    [ -d "$d" ] || continue
    label="$(basename "$d")"; label="${label#${PREFIX}_}"
    case " $LABELS " in *" $label "*) continue ;; esac
    STALE="$STALE $d"
done
if [ -n "$STALE" ]; then
    echo "[sweep] 接頭辞 $PREFIX に、今回の条件に無いディレクトリが残っている:"
    for d in $STALE; do echo "    $d"; done
    echo
    echo "        結果表は results/${PREFIX}_* を全部拾うので、前のラウンドの"
    echo "        行が今回の表に混ざる。土台が違えば比較にならない。"
    echo "        消すか:"
    # shellcheck disable=SC2086
    echo "            rm -rf$STALE"
    echo "        別の接頭辞にすること:"
    echo "            PARC_SWEEP_PREFIX=${PREFIX}b ..."
    exit 1
fi

# --- 多重起動の防止 ---------------------------------------------------------
# sweep_nexec.sh と同じロックを使う。別種のスイープでも、同時に走れば同じ
# ポートと同じ出力先を奪い合う。先発だけがポートを握り、後発のサーバーは
# bind に失敗して死ぬ一方で /health は先発が応答するため、全条件が同じ
# サーバーに対して評価され、全部同じ結果になる。
LOCK="$ROOT/.sweep.lock"
PIDFILE="$ROOT/.sweep.pid"
exec 9>>"$LOCK" || true          # >> にする。> だと保持者の PID を消してしまう
if command -v flock >/dev/null 2>&1; then
    # PARC_SWEEP_WAIT=1 なら、走っているラウンドが終わるまで待って続きを始める。
    #
    # 既定は即中止だが、そのせいで「終わったか確かめて、空いていたら起動する」
    # を手で繰り返すことになる。確認と起動を続けて貼ると、確認の結果を読む前に
    # 起動が走って弾かれる（実際に 2 回続けて起きた）。並べて投げておけるほうが
    # 安全である。
    if [ "${PARC_SWEEP_WAIT:-0}" != "0" ] && ! flock -n 9; then
        holder="$(cat "$PIDFILE" 2>/dev/null || true)"
        echo "[sweep] 別のスイープ (pid=${holder:-?}) が動いている。空くまで待つ。"
        echo "        待たずに中止させるには PARC_SWEEP_WAIT を外すこと。"
        flock 9
        echo "[sweep] ロックを取得した。開始する。"
    fi
    if ! flock -n 9; then
        holder="$(cat "$PIDFILE" 2>/dev/null || true)"
        echo "[sweep] 既に別のスイープが動いている。中止する。"
        echo "        終わり次第これを始めたいなら: PARC_SWEEP_WAIT=1 を付ける"
        if [ -n "$holder" ] && kill -0 "$holder" 2>/dev/null; then
            echo "    pid=$holder  $(tr '\0' ' ' < "/proc/$holder/cmdline" 2>/dev/null | cut -c1-90)"
            # プロセスグループごと送る。`kill $holder` だと bash 本体にしか
            # 届かず、bash は前景の `python -m pipeline` が終わるまで trap を
            # 走らせないので、その条件（数十分）が終わるまで止まらない。
            echo "        止めるには: kill -TERM -$holder    # 先頭の - はプロセスグループ"
            echo "        （評価中の python ごと落とす。kill $holder だと条件の終わりまで効かない）"
        else
            echo "        止めるには: pkill -f sweep_"
        fi
        exit 1
    fi
    echo "$$" > "$PIDFILE"
fi

# --- コンソール出力の控えを自分で持つ ---------------------------------------
# 呼び出し側は普通 `> logs/sweep_<接頭辞>.log` で回すが、走行中に同じ行を
# もう一度叩くと（起動できたか確かめたくなる）、その `>` が**走行中のログを
# 切り詰める**。2 回やっている。中止されるのは後発なのに、記録が消えるのは
# 先発のほうなので、気づくのが遅れる。
#
# ロックを取れた側だけが、自分の PID 付きの控えへ複製する。呼び出し側の
# リダイレクト先が潰されても、こちらは残る。
ROUND_LOG="logs/sweep_${PREFIX}_$$.log"
exec > >(tee -a "$ROUND_LOG") 2>&1
echo "[sweep] このラウンドの控え: $ROUND_LOG"

SRV=""
ROUND_SIG=""          # このラウンドが読んだモデル。条件をまたいで変わってはいけない
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
    OUT="results/${PREFIX}_${label}"
    SRVLOG="logs/server_${PREFIX}_${label}.log"
    echo
    echo "=== $label : $(cond_desc "$assign") ==============================="
    rm -rf "$OUT"; mkdir -p "$OUT"

    require_port_free || exit 1
    # shellcheck disable=SC2086
    env "${CLEAR[@]}" ${COMMON//,/ } ${assign//,/ } PARC_PORT="$PORT" \
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
    #
    # 拾う行を間違えないこと。run_policy_server.sh が先に
    #   [run] ensemble=1  h=8  query=1(既定)  m=0.01(既定)
    # を出すので、`grep -o 'ensemble=.*' | head -1` はそちらに当たる。この行は
    # 「シェルが何を渡したか」であって「サーバーが何で立ったか」ではないうえ、
    # 値が `1` なので下の off/on どちらのパターンにも一致せず、**検査が常に
    # 素通りしていた**。誤検知が出ないので、壊れていることに気づけなかった。
    # サーバー自身が出す [MyPolicy] の行だけを見る。
    ens_line="$(grep -oE '^\[MyPolicy\][[:space:]]+ensemble=.*' "$SRVLOG" \
                | head -1 | sed 's/.*ensemble=/ensemble=/')"
    # 判定は共通部と条件を合わせた「実際に効く設定」に対して行う。条件側だけ
    # を見ると、PARC_SWEEP_COMMON で ensembling を掛けたラウンドの対照条件が
    # 「つまみ無しのはずなのに on」と誤検知される。
    case "$COMMON,$assign" in
        *PARC_ENSEMBLE=1*)
            if [ "${ens_line#ensemble=off}" != "$ens_line" ]; then
                echo "[sweep] $label は ensembling を要求したのにサーバーは off で起動した。"
                echo "        起動ログ: $ens_line"
                cleanup; SRV=""; exit 1
            fi ;;
        *)
            if [ "${ens_line#ensemble=on}" != "$ens_line" ]; then
                echo "[sweep] $label は ensembling を要求していないのに on で起動した。"
                echo "        起動ログ: $ens_line  （シェルに PARC_ENSEMBLE が残っていないか）"
                cleanup; SRV=""; exit 1
            fi ;;
    esac

    # 全条件が同じモデル・同じ重みを読んだことを確かめる。ラウンドの途中で
    # 混ざると、表は普通に出るのに比較になっていない（つまみの効果とモデルの
    # 差が混ざる）。起動ログでしか検出できない。
    sig="$(grep -oE '^\[MyPolicy\] [^ ]+ ready' "$SRVLOG" | head -1 | awk '{print $2}')"
    sig="$sig@$(grep -oE '^\[MyPolicy\] weights: [^ ]+' "$SRVLOG" | head -1 | awk '{print $3}')"
    if [ -z "$ROUND_SIG" ]; then
        ROUND_SIG="$sig"
        echo "[sweep] このラウンドのモデル: $sig"
    elif [ "$sig" != "$ROUND_SIG" ]; then
        echo "[sweep] $label は他の条件と違うモデルを読んでいる。"
        echo "        最初の条件: $ROUND_SIG"
        echo "        この条件  : $sig"
        echo "        比較にならないので中止する。"
        cleanup; SRV=""; exit 1
    fi

    python -m pipeline --server-url "http://127.0.0.1:$PORT" --track track1 \
        ${TASK_ARGS[@]+"${TASK_ARGS[@]}"} \
        --n-episodes "$EPISODES" --max-steps "$MAX_STEPS" \
        --timeout 10 --seed "$SEED" --output-dir "$OUT" \
        > "logs/eval_${PREFIX}_${label}.log" 2>&1
    rc=$?
    [ "$rc" = "0" ] || echo "[sweep] 評価が rc=$rc で終了。logs/eval_${PREFIX}_${label}.log を確認"

    # レイテンシは必ず見る。1 リクエストでも 10 秒を超えるとトラックが 0 点になる。
    grep 'レイテンシ' "$SRVLOG" | tail -3 | sed 's/^/    /'
    # |Δaction| は slew の上限を選ぶための材料。制限が無効でも出る。
    # 内訳（xyz / rot）は続く 2 行なので -A 2。見出し行だけ拾っても意味が無い。
    grep -A 2 'Δaction' "$SRVLOG" | tail -9 | sed 's/^/    /'
    # 指の開きは把持失敗の判定閾値を選ぶための材料。これも制限が無効でも出る。
    # レイテンシ行と並べる（そちらの n= が step 数 = 成否そのもの）。
    grep -E 'レイテンシ|gripper:' "$SRVLOG" | tail -8 | sed 's/^/    /'

    cleanup; SRV=""
    for _ in $(seq 1 30); do health_ok || break; sleep 1; done
done

echo
echo "======================================================"
echo " 結果"
echo "======================================================"
python3 tools/show_ensemble_results.py --prefix "$PREFIX" --episodes "$EPISODES"

echo
echo "採用するなら submission/policy_server.py の TEMPORAL_ENSEMBLE / ENSEMBLE_* の"
echo "既定と、ENVIRONMENT_SETUP.md §29 の表を書き換えること。"
