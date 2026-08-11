#!/usr/bin/env bash
#
# 走行中のスイープの進捗をまとめて出す。読み取り専用で、評価には触らない。
#
#   bash tools/sweep_status.sh             # 走行中のラウンドを自動判定
#   bash tools/sweep_status.sh t4 10       # results/t4_*/ のラウンド（10 ep）
#   watch -n 60 bash tools/sweep_status.sh t4 10   # 1 分ごとに更新
#
# 第 2 引数は 1 条件あたりのエピソード数。標準誤差の表示に使うだけだが、
# ここを間違えると「読んではいけない差」を読んでしまうので合わせること。
#
# 3 段階で見る。
#   1. どの条件を走っているか        … スイープ本体のログ
#   2. その条件で何タスク終わったか  … 評価ログの「タスク評価完了」
#   3. 終わった条件の成績            … show_ensemble_results.py
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 接頭辞を省略したら、**いま走っているスイープ**のものを使う。
#
# 既定を "ens" に固定していたせいで、別ラウンドを回している最中に
# 何ヶ月も前のラウンドの成績を表示し、それを現在の結果として読みかけた。
# 2 回やっている。走っているものがあるならそれを見るのが正しい。
detect_prefix() {
    # スイープ本体は「出力 : results/<接頭辞>_<ラベル>/」を冒頭に出す。
    #
    # ポートを取れずに即中止したログは飛ばす。中止されたラウンドのほうが
    # 新しいことは普通にあり（走行中のものを追い越して起動しようとした結果
    # 落ちるので）、それを拾うと走っていないラウンドを見せてしまう。
    local log
    for log in $(ls -t logs/sweep_*.log 2>/dev/null); do
        grep -q '既に別のスイープが動いている' "$log" && continue
        SWEEP_SRC_LOG="$log"
        sed -n 's#.*results/\([A-Za-z0-9_.-]*\)_<ラベル>/.*#\1#p' "$log" | head -1
        return 0
    done
    return 1
}

PREFIX="${1:-}"
if [ -z "$PREFIX" ]; then
    # サブシェルで呼ぶと SWEEP_SRC_LOG が親に残らないので、いったんファイルへ
    PREFIX="$(detect_prefix > /tmp/.parc_prefix.$$ 2>/dev/null; cat /tmp/.parc_prefix.$$ 2>/dev/null; rm -f /tmp/.parc_prefix.$$)"
    SWEEP_SRC_LOG="$(for log in $(ls -t logs/sweep_*.log 2>/dev/null); do
        grep -q '既に別のスイープが動いている' "$log" && continue
        echo "$log"; break
    done)"
    if [ -n "$PREFIX" ]; then
        echo "[status] 接頭辞を最新のスイープログから判定: $PREFIX"
    else
        PREFIX="ens"
        echo "[status] スイープログが無いので既定の接頭辞を使う: $PREFIX"
    fi
fi
# エピソード数も同じログから拾う。既定の 50 のままだと標準誤差の表示が
# 実際より小さく出て、**読んではいけない差を読める差として見せてしまう**。
# 10 ep x 4 タスクなら 6.3 pt なのに 2.8 pt と表示された実績がある。
EPISODES="${2:-${PARC_SWEEP_EPISODES:-}}"
if [ -z "$EPISODES" ] && [ -n "${SWEEP_SRC_LOG:-}" ]; then
    EPISODES="$(sed -n 's#.*エピソード  : \([0-9][0-9]*\) / 条件.*#\1#p' "$SWEEP_SRC_LOG" | head -1)"
    [ -n "$EPISODES" ] && echo "[status] エピソード数をログから判定: $EPISODES / 条件"
fi
EPISODES="${EPISODES:-50}"

echo "=== プロセス ==============================================="
if pgrep -af 'sweep_ensemble\.sh' >/dev/null 2>&1; then
    pgrep -af 'sweep_ensemble\.sh' | sed 's/^/  /'
else
    echo "  スイープは走っていない（完走したか、中断された）"
fi
pgrep -af 'policy_server\.py' 2>/dev/null | sed 's/^/  /'

echo
echo "=== いまどの条件か ========================================="
# 名前は 2 通りある。nohup の出力先を自分で決めると sweep_<接頭辞>.log に
# なり、スクリプトが自分で作ると sweep_<接頭辞>_<pid>.log になる。両方見る。
SWEEPLOG="$(ls -t logs/sweep_${PREFIX}_*.log logs/sweep_${PREFIX}.log 2>/dev/null | head -1)"
if [ -n "$SWEEPLOG" ]; then
    echo "  ログ: $SWEEPLOG"
    grep -E '^=== |^\[sweep\]' "$SWEEPLOG" | tail -6 | sed 's/^/  /'
else
    echo "  logs/sweep_${PREFIX}{,_*}.log が無い"
fi

echo
echo "=== 条件ごとの進み具合 ====================================="
shopt -s nullglob
found=0
for log in logs/eval_${PREFIX}_*.log; do
    found=1
    label="${log#logs/eval_${PREFIX}_}"; label="${label%.log}"
    # grep -c は 0 件でも "0" を出して exit 1 する。|| echo 0 を付けると
    # "0" が 2 行になるので付けない（set -e は無いので失敗しても止まらない）。
    done_n="$(grep -c 'タスク評価完了' "$log" 2>/dev/null)"
    # いま走っているエピソードの進み（最後の [進捗] 行）
    last="$(grep '\[進捗\]' "$log" 2>/dev/null | tail -1 | sed 's/.*\[進捗\] *//')"
    printf '  %-8s タスク完了 %s   %s\n' "$label" "$done_n" "${last:-—}"
    grep 'タスク評価完了' "$log" 2>/dev/null \
        | sed 's/.*タスク評価完了: /      /' | cut -c1-100
done
[ "$found" = "1" ] || echo "  logs/eval_${PREFIX}_*.log が無い"

echo
echo "=== レイテンシ（10 秒を 1 回でも超えるとトラックが 0 点）==="
for log in logs/server_${PREFIX}_*.log; do
    label="${log#logs/server_${PREFIX}_}"; label="${label%.log}"
    line="$(grep 'レイテンシ' "$log" 2>/dev/null | tail -1 | sed 's/.*レイテンシ: //')"
    slow="$(grep -c '遅い /act' "$log" 2>/dev/null)"
    printf '  %-8s %s  [遅延警告 %s 回]\n' "$label" "${line:-—}" "$slow"
done

echo
echo "=== 終わった条件の成績 ====================================="
python3 tools/show_ensemble_results.py --prefix "$PREFIX" --episodes "$EPISODES"
