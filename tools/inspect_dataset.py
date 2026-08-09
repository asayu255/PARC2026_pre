#!/usr/bin/env python
"""lerobot/libero_plus のメタデータを読み、LoRA 学習用のエピソードを選ぶ。

メタデータ（meta/*）だけを扱う。動画は一切ダウンロードしない。

    # 構成を見る
    python tools/inspect_dataset.py

    # 摂動グループの推定（front カメラの画像統計をクラスタリング）
    python tools/inspect_dataset.py --cluster

    # 学習用エピソードを選んで JSON に書く
    python tools/inspect_dataset.py --select 60 --out episodes.json

lerobot 0.4.4 には `lerobot.datasets.dataset_metadata` が無い
（examples/ のノートブックは別バージョンの lerobot 向け）。
そのため parquet を直接読む。この方が版に依存しない。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pyarrow.parquet as pq

DATASET_REPO = "lerobot/libero_plus"
DATASET_REVISION = "f3f49f426d75030177b18778374005bc12ccd588"

# 摂動の手掛かりに使う列。front カメラの明るさと分散は
# 背景テクスチャ・照明条件の摂動で動く。
STAT_COLUMNS = (
    "stats/observation.images.front/mean",
    "stats/observation.images.front/std",
)


def meta_root() -> str:
    """meta/* だけを取得したスナップショットのパスを返す。"""
    from huggingface_hub import snapshot_download

    return snapshot_download(
        DATASET_REPO,
        repo_type="dataset",
        revision=DATASET_REVISION,
        allow_patterns=["meta/*"],
    )


def flat(value) -> list[float]:
    """入れ子リスト（list<list<list<double>>> 等）を平坦な float 列にする。"""
    if value is None:
        return []
    if isinstance(value, (float, int, np.floating, np.integer)):
        return [float(value)]
    out: list[float] = []
    for item in value:
        out.extend(flat(item))
    return out


def task_name(cell) -> str:
    """tasks 列は list<string>。先頭を採る。"""
    if isinstance(cell, str):
        return cell
    try:
        return str(cell[0])
    except (TypeError, IndexError, KeyError):
        return str(cell)


def load_episodes(root: str, with_stats: bool):
    path = glob.glob(
        os.path.join(root, "meta", "episodes", "**", "*.parquet"), recursive=True
    )
    if not path:
        raise SystemExit(f"episodes parquet が見つからない: {root}/meta/episodes")

    columns = ["episode_index", "tasks", "length"]
    if with_stats:
        available = set(pq.ParquetFile(path[0]).schema_arrow.names)
        columns += [c for c in STAT_COLUMNS if c in available]

    return pq.read_table(path, columns=columns).to_pandas()


def print_info(root: str) -> None:
    info_path = os.path.join(root, "meta", "info.json")
    if not os.path.isfile(info_path):
        return
    info = json.load(open(info_path))
    print("=== info.json ===")
    for key in (
        "codebase_version",
        "total_episodes",
        "total_frames",
        "total_tasks",
        "fps",
    ):
        if key in info:
            print(f"  {key}: {info[key]}")
    print()


def print_tasks(df) -> dict[str, list[int]]:
    by_task: dict[str, list[int]] = defaultdict(list)
    for episode_index, cell in zip(df["episode_index"], df["tasks"]):
        by_task[task_name(cell)].append(int(episode_index))

    lengths = np.asarray(df["length"], dtype=float)
    print("=== エピソード ===")
    print(f"  タスク数: {len(by_task)}   エピソード: {len(df)}")
    print(
        f"  length: 平均 {lengths.mean():.1f}  中央 {np.median(lengths):.0f}  "
        f"最小 {lengths.min():.0f}  最大 {lengths.max():.0f}"
    )
    counts = sorted(len(v) for v in by_task.values())
    print(f"  タスクあたり: 最小 {counts[0]}  中央 {counts[len(counts) // 2]}  最大 {counts[-1]}")
    print()
    return by_task


def print_clusters(df, by_task: dict[str, list[int]], n_show: int) -> None:
    """front カメラの画像統計から摂動の多様性を推定する。

    摂動が離散的な条件（背景 N 種 × 照明 M 種）なら、
    同一タスク内でも平均輝度が広く散る。散らないなら
    データセットに摂動バリアントは入っていない。
    """
    mean_col = "stats/observation.images.front/mean"
    std_col = "stats/observation.images.front/std"
    if mean_col not in df.columns:
        print("画像統計の列が無いのでクラスタ推定は省略。")
        return

    brightness: dict[int, float] = {}
    spread: dict[int, float] = {}
    for episode_index, mean_cell, std_cell in zip(
        df["episode_index"], df[mean_col], df.get(std_col, df[mean_col])
    ):
        m = flat(mean_cell)
        s = flat(std_cell)
        if m:
            brightness[int(episode_index)] = float(np.mean(m))
        if s:
            spread[int(episode_index)] = float(np.mean(s))

    values = np.asarray(list(brightness.values()))
    print("=== front カメラの平均輝度（摂動の代理指標）===")
    print(
        f"  全体: 平均 {values.mean():.4f}  std {values.std():.4f}  "
        f"範囲 {values.min():.4f}..{values.max():.4f}"
    )

    rows = []
    for name, episodes in by_task.items():
        vals = np.asarray([brightness[e] for e in episodes if e in brightness])
        if vals.size:
            rows.append((float(vals.std()), float(vals.min()), float(vals.max()), len(vals), name))
    rows.sort(reverse=True)

    print("\n  タスク内のばらつきが大きい順:")
    print(f"  {'std':>8} {'min':>8} {'max':>8} {'本数':>6}  タスク")
    for std, lo, hi, n, name in rows[:n_show]:
        print(f"  {std:8.4f} {lo:8.4f} {hi:8.4f} {n:6d}  {name[:60]}")

    within = float(np.mean([r[0] for r in rows]))
    print(f"\n  タスク内 std の平均: {within:.4f}   全体 std: {values.std():.4f}")
    if within > 0.5 * values.std():
        print("  -> タスク内のばらつきが全体と同程度。摂動バリアントが含まれている。")
        print("     エピソードを広く取ることが摂動網羅に直結する。")
    else:
        print("  -> タスク内のばらつきが小さい。摂動バリアントは入っていないか、")
        print("     輝度に出ない種類（物体位置・カメラ姿勢など）に限られる。")
        print("     その場合は image_transforms によるオーグメンテーションの比重が上がる。")
    print()


def choose_evenly_spaced(indices: list[int], count: int) -> list[int]:
    """episode_index 順に均等間隔で選ぶ。

    このデータセットはタスクを round-robin で書き出しており
    （episode 0=task0, 1=task1, ...）、摂動条件も index 方向に
    並んでいる可能性が高い。連続塊ではなく均等間隔で取る。
    """
    ordered = sorted(indices)
    if count >= len(ordered):
        return ordered
    if count == 1:
        return [ordered[len(ordered) // 2]]
    positions = [round(i * (len(ordered) - 1) / (count - 1)) for i in range(count)]
    return sorted({ordered[p] for p in positions})


def select(df, by_task: dict[str, list[int]], per_task: int, out_path: str | None) -> None:
    length_of = {
        int(e): int(n) for e, n in zip(df["episode_index"], df["length"])
    }

    selected: list[int] = []
    for name in sorted(by_task):
        selected.extend(choose_evenly_spaced(by_task[name], per_task))
    selected.sort()

    frames = sum(length_of.get(e, 0) for e in selected)
    print("=== 選択 ===")
    print(f"  タスクあたり {per_task} 本  ->  {len(selected)} エピソード / {frames} フレーム")
    for batch in (16, 32, 64):
        print(f"  batch {batch:3d}: 1 epoch = {frames // batch} steps")

    if out_path:
        # lerobot-train の --dataset.episodes に丸ごと渡すので空白を入れない。
        with open(out_path, "w") as fh:
            json.dump(selected, fh, separators=(",", ":"))
        print(f"  書き出し: {out_path}")
    else:
        print("  --out を付けると JSON に書き出す（lerobot-train の --dataset.episodes 用）")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="front カメラの画像統計から摂動の多様性を推定する",
    )
    parser.add_argument(
        "--select",
        type=int,
        metavar="N",
        help="タスクあたり N 本を均等間隔で選ぶ",
    )
    parser.add_argument("--out", help="--select の結果を書き出す JSON のパス")
    parser.add_argument(
        "--top", type=int, default=12, help="--cluster で表示するタスク数"
    )
    args = parser.parse_args()

    root = meta_root()
    print(f"meta: {root}\n")
    print_info(root)

    df = load_episodes(root, with_stats=args.cluster)
    by_task = print_tasks(df)

    if args.cluster:
        print_clusters(df, by_task, args.top)

    if args.select:
        select(df, by_task, args.select, args.out)


if __name__ == "__main__":
    main()
