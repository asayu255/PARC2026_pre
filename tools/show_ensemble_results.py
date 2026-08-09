#!/usr/bin/env python3
"""results/ens_*/ の結果を読んで比較表を出す。

sweep_ensemble.sh は完走時に同じ表をログの末尾へ出すが、こちらは
いつでも何度でも出せる。途中経過の確認、ログを流してしまったとき、
条件を追加で回したあとの再集計に使う。

    python3 tools/show_ensemble_results.py
    python3 tools/show_ensemble_results.py --prefix nexec   # results/nexec*/ を見る

判定の指針（§28.1 / §29.5）:
  主指標は collision と jerk である。成功率ではない。同一条件の再測定で
  success が 0.92 -> 0.80、collision が 0.080 -> 0.180 と動いた実績が
  あるので、その幅に収まる差は読まないこと。
"""
import argparse
import glob
import json
import math
import os

METRICS = [
    ("success_rate", "success", None),
    ("collision_rate", "collision", "metrics"),
    ("cartesian_path_length", "cartesian", "metrics"),
    ("rms_cartesian_jerk", "jerk(rms)", "metrics"),
    ("sparc", "sparc", "metrics"),
]


def load(directory):
    """その条件の最新の結果 JSON から task 1 件ぶんを返す。無ければ None。"""
    files = sorted(glob.glob(os.path.join(directory, "*.json")))
    if not files:
        return None
    with open(files[-1]) as fh:
        data = json.load(fh)
    tracks = data.get("tracks") or []
    tasks = (tracks[0].get("tasks") if tracks else []) or []
    return tasks[0] if tasks else None


def value(task, key, where):
    v = (task.get("metrics", {}) if where else task).get(key)
    return v if isinstance(v, (int, float)) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="ens",
                    help="results/<prefix>_*/ を集計する（既定: ens）")
    ap.add_argument("--baseline", default="base",
                    help="差分の基準にする条件名（既定: base）")
    args = ap.parse_args()

    dirs = sorted(glob.glob(f"results/{args.prefix}_*/"))
    if not dirs:
        print(f"results/{args.prefix}_*/ が無い。まだ 1 条件も終わっていない。")
        return

    rows = []
    for d in dirs:
        label = os.path.basename(d.rstrip("/"))[len(args.prefix) + 1:]
        rows.append((label, load(d)))

    header = f"{'条件':>8}" + "".join(f"{name:>11}" for _, name, _ in METRICS)
    print(header)
    for label, task in rows:
        if task is None:
            print(f"{label:>8}   --- まだ結果なし ---")
            continue
        cells = []
        for key, _, where in METRICS:
            v = value(task, key, where)
            cells.append(f"{v:>11.3f}" if v is not None else f"{'-':>11}")
        print(f"{label:>8}" + "".join(cells))

    base = dict(rows).get(args.baseline)
    if base is not None and len(rows) > 1:
        print(f"\n{args.baseline} との差:")
        for label, task in rows:
            if task is None or label == args.baseline:
                continue
            parts = []
            for key, name, where in METRICS:
                if name == "cartesian" or name == "sparc":
                    continue
                a, b = value(task, key, where), value(base, key, where)
                if a is not None and b is not None:
                    parts.append(f"{name} {a - b:+.3f}")
            print(f"  {label:>8}  " + "  ".join(parts))

    done = [t for _, t in rows if t is not None]
    if done:
        n = 50
        se = math.sqrt(0.2 * 0.8 / n)
        print()
        print(f"success の標準誤差は p=0.8・{n} エピソードで約 {se * 100:.1f} pt。")
        print("同一条件の再測定でも collision は 0.080 -> 0.180 動いた実績がある。")
        print("jerk（分散が小さく、独立な測定で再現している）を主に見ること。")


if __name__ == "__main__":
    main()
