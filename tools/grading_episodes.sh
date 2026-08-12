#!/usr/bin/env bash
#
# 採点ログからエピソードごとの step 数を復元し、スコアを分解する。
#
#   bash tools/grading_episodes.sh logs/grade_tta2.log
#   bash tools/grading_episodes.sh logs/grade_tta2.log 0.304
#
# 採点結果として返ってくるのは合計スコア 1 個だけだが、ログの
# `POST /reset` と `POST /act` を数えれば 8 本それぞれの step 数が復元できる。
#
#   300 ちょうど -> 時間切れ = 失敗
#   300 未満     -> その手数で成功
#
# サーバー自身が出す「前エピソードのレイテンシ」は `/reset` のたびに**前の**
# エピソードを報告するので、**最後の 1 本は必ず出ない**。こちらは /act を
# 直接数えるので 8 本すべて取れる。
#
# 採点式の作業仮説（Slack 由来・運営未確認、§5）:
#
#   score = Σ(成功したエピソードの smooth) / 8
#
# smooth は「滑らかさ・実行効率・安全性」の合成で、失敗エピソードの寄与は
# ゼロ。第 2 引数にスコアを渡すと 1 本あたりの smooth を逆算する。
# 逆算値が 1 を超えたらこの仮説は成り立っていない（そのことも表示する）。
set -uo pipefail

LOG="${1:-}"
SCORE="${2:-}"
MAX_STEPS="${PARC_GRADE_MAX_STEPS:-300}"

if [ -z "$LOG" ]; then
    echo 'usage: bash tools/grading_episodes.sh <採点ログ> [スコア]' >&2
    echo '  例:  bash tools/grading_episodes.sh logs/grade_tta3.log 0.29' >&2
    exit 2
fi
[ -f "$LOG" ] || { echo "ERROR: $LOG が無い" >&2; exit 1; }

# grep -a: 二重起動でログが切り詰められると NUL が混ざり binary 扱いになる
STEPS="$(grep -aoE 'POST /(act|reset)' "$LOG" \
    | awk '/reset/{if(n!="")print n; n=0; next} {n++} END{if(n!="")print n}')"

if [ -z "$STEPS" ]; then
    echo "ERROR: $LOG に POST /act も POST /reset も無い。採点ログか確認すること。" >&2
    exit 1
fi

echo "=== エピソードごとの step 数 ==="
i=0; ok=0; sum_ok=0
for n in $STEPS; do
    i=$((i + 1))
    if [ "$n" -lt "$MAX_STEPS" ]; then
        printf '  ep%-2d %4d steps   成功\n' "$i" "$n"
        ok=$((ok + 1)); sum_ok=$((sum_ok + n))
    else
        printf '  ep%-2d %4d steps   時間切れ\n' "$i" "$n"
    fi
done

echo
echo "  エピソード数 : $i"
echo "  成功         : $ok / $i"
[ "$ok" -gt 0 ] && echo "  成功の平均長 : $((sum_ok / ok)) steps"

if [ -n "$SCORE" ]; then
    echo
    echo "=== スコアの分解（作業仮説 score = Σsmooth / $i）==="
    awk -v s="$SCORE" -v n="$i" -v k="$ok" 'BEGIN{
        tot = s * n;
        printf "  Σsmooth      : %.3f\n", tot;
        if (k == 0) { print "  成功 0 本なのにスコアが出ている -> 仮説が誤り"; exit }
        printf "  1 本あたり   : %.3f\n", tot / k;
        if (tot / k > 1.0)
            printf "  ** 1 を超えた。成功本数がこれより多いか、仮説が誤り **\n";
        printf "  この本数での上限: %.3f  (= %d/%d、smooth=1 のとき)\n", k / n, k, n;
        printf "  上限に対する達成率: %.0f%%\n", 100 * s / (k / n);
    }'
fi
