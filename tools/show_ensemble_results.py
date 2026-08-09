#!/usr/bin/env python3
"""results/<接頭辞>_*/ の結果を読んで比較表を出す。

sweep_ensemble.sh が完走時に呼ぶのもこれである。単体でも動くので、
途中経過の確認、ログを流してしまったとき、条件を追加で回したあとの
再集計に使える。

    python3 tools/show_ensemble_results.py                    # results/ens_*/
    python3 tools/show_ensemble_results.py --prefix t4        # 4 タスク回帰のラウンド
    python3 tools/show_ensemble_results.py --prefix nexec     # §25.1 の古いスイープ

複数タスクを回したラウンドでは `overall_metrics`（タスク平均）を読む。
単一タスクのラウンドではそれがタスク自身の値と一致するので、どちらも
同じ表になる。
"""
import argparse
import glob
import json
import math
import os

# (JSON のキー, 表示名) — overall_metrics 側は mean_ が付く
METRICS = [
    ("success_rate", "success"),
    ("collision_rate", "collision"),
    ("cartesian_path_length", "cartesian"),
    ("rms_cartesian_jerk", "jerk"),
    ("sparc", "sparc"),
]

# 差分を出す指標。cartesian と sparc は判定に使わないので省く。
DIFF_METRICS = ["success", "collision", "jerk"]


def load(directory):
    """その条件の最新の結果 JSON から {表示名: 値} と タスク数 を返す。"""
    files = sorted(glob.glob(os.path.join(directory, "*.json")))
    if not files:
        return None, 0
    with open(files[-1]) as fh:
        data = json.load(fh)
    tracks = data.get("tracks") or []
    if not tracks:
        return None, 0
    track = tracks[0]
    tasks = track.get("tasks") or []
    if not tasks:
        return None, 0

    overall = track.get("overall_metrics") or {}
    first = tasks[0]
    first_metrics = first.get("metrics", {})

    out = {}
    for key, name in METRICS:
        # タスク平均を優先する。無い古い JSON では 1 タスク目から拾う。
        v = overall.get(f"mean_{key}")
        if not isinstance(v, (int, float)):
            v = first.get(key) if key == "success_rate" else first_metrics.get(key)
        out[name] = v if isinstance(v, (int, float)) else None
    return out, len(tasks)


def per_task(directory):
    """タスクごとの (名前, success_rate, collision_rate) を返す。"""
    files = sorted(glob.glob(os.path.join(directory, "*.json")))
    if not files:
        return []
    with open(files[-1]) as fh:
        data = json.load(fh)
    tracks = data.get("tracks") or []
    out = []
    for task in (tracks[0].get("tasks") if tracks else []) or []:
        out.append((task.get("task_name", "?"),
                    task.get("success_rate"),
                    task.get("metrics", {}).get("collision_rate")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="ens",
                    help="results/<prefix>_*/ を集計する（既定: ens）")
    ap.add_argument("--baseline", default="base",
                    help="差分の基準にする条件名（既定: base）")
    ap.add_argument("--episodes", type=int, default=50,
                    help="1 条件あたりのエピソード数。標準誤差の表示に使う")
    ap.add_argument("--per-task", action="store_true",
                    help="タスクごとの success / collision も出す")
    args = ap.parse_args()

    dirs = sorted(glob.glob(f"results/{args.prefix}_*/"))
    if not dirs:
        print(f"results/{args.prefix}_*/ が無い。まだ 1 条件も終わっていない。")
        return

    rows, n_tasks = [], 0
    for d in dirs:
        label = os.path.basename(d.rstrip("/"))[len(args.prefix) + 1:]
        values, n = load(d)
        n_tasks = max(n_tasks, n)
        rows.append((label, values))

    print(f"{'条件':>8}" + "".join(f"{name:>11}" for _, name in METRICS))
    for label, values in rows:
        if values is None:
            print(f"{label:>8}   --- まだ結果なし ---")
            continue
        cells = "".join(
            f"{values[name]:>11.3f}" if values[name] is not None else f"{'-':>11}"
            for _, name in METRICS
        )
        print(f"{label:>8}{cells}")

    if args.per_task:
        print()
        print("タスク別:")
        per = {}
        for d in dirs:
            label = os.path.basename(d.rstrip("/"))[len(args.prefix) + 1:]
            for name, sr, col in per_task(d):
                per.setdefault(name, {})[label] = (sr, col)
        labels = [lbl for lbl, v in rows if v is not None]
        print(f"  {'タスク':<46}" + "".join(f"{lbl:>18}" for lbl in labels))
        for name, by_label in per.items():
            cells = ""
            for lbl in labels:
                sr, col = by_label.get(lbl, (None, None))
                cells += (f"{sr:>10.2f}/{col:<7.2f}" if sr is not None else f"{'-':>18}")
            print(f"  {name[:46]:<46}" + cells)
        print("  （success / collision）")

    base = dict(rows).get(args.baseline)
    if base:
        print(f"\n{args.baseline} との差:")
        for label, values in rows:
            if not values or label == args.baseline:
                continue
            parts = []
            for name in DIFF_METRICS:
                a, b = values[name], base[name]
                if a is None or b is None:
                    continue
                pct = f" ({(a - b) / b * 100:+.0f}%)" if name == "jerk" and b else ""
                parts.append(f"{name} {a - b:+.3f}{pct}")
            print(f"  {label:>8}  " + "  ".join(parts))

    total = args.episodes * max(1, n_tasks)
    se = math.sqrt(0.2 * 0.8 / total) * 100
    print()
    print("読み方:")
    print(f"  エピソード数は 1 条件 {args.episodes} x {n_tasks} タスク = {total} 本。"
          f" success の標準誤差は約 {se:.1f} pt。")
    print("  同一条件の再測定でも collision は 0.080 -> 0.180 動いた実績がある")
    print("  （2026-08-09、stove 50 本 x 2 回）。この幅の差は読まないこと。")
    print()
    print("  信用できるのは jerk である。1 エピソード約 90 step x 本数の平均で")
    print("  分散が小さく、同一条件の再測定でのぶれは 2.6% だった。")
    print("  jerk が 10% を超えて動いていれば実効果と見てよい。")
    print()
    print("  jerk が下がる      -> チャンク境界の不連続が減っている。狙いどおり")
    print("  collision が下がる -> 採点スコアに直接効く（採点で到達した 2 本は")
    print("                        どちらも 1mm ルールで落ちている。§22.3.1）")
    print("  どれも動かない     -> この軸も閉じる。§27.3 の追加学習へ戻る")


if __name__ == "__main__":
    main()
