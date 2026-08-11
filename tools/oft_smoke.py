"""OpenVLA-OFT+ が載って動くかを、ポリシーサーバー抜きで確かめる。

    PARC_OFT_WEIGHTS=~/parc_models/oft_libero_plus \
      ~/miniforge3/envs/parc-oft/bin/python tools/oft_smoke.py

見たいのは 4 つ。

  1. オフラインで（HF に触らずに）ロードできるか
  2. VRAM をどれだけ食うか。採点機の GPU は未知で、いまの SmolVLA は 450M なので
     一度も試していない。15 GB の bf16 が載らなければ提出枠を捨てることになる
  3. 1 リクエストのレイテンシ。制限は 10 秒
  4. action がまともな範囲に出るか。壊れていれば飽和するか NaN になる

--iters で回数を変えられる。1 回目はコンパイル等が入るので、2 回目以降を見る。
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "submission"))

import oft_policy  # noqa: E402


def fake_obs(rng: np.random.Generator, size: int = 128):
    """PARC が実際に送ってくる形（128x128 uint8 + 8 次元 state）を模す。"""
    return {
        "image": rng.integers(0, 256, (size, size, 3), dtype=np.uint8),
        "wrist": rng.integers(0, 256, (size, size, 3), dtype=np.uint8),
        # eef_pos(3) + axis_angle(3) + gripper_qpos(2)
        "state": np.array([-0.05, 0.02, 1.02, 3.10, -0.05, 0.02, 0.03, -0.03]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.environ.get("PARC_OFT_WEIGHTS", ""))
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--task", default="pick up the black bowl and place it on the plate")
    ap.add_argument("--no-center-crop", action="store_true")
    ap.add_argument("--no-flip", action="store_true")
    args = ap.parse_args()

    if not args.weights:
        print("PARC_OFT_WEIGHTS か --weights で checkpoint を指すこと。", file=sys.stderr)
        return 2

    import torch

    t0 = time.perf_counter()
    model = oft_policy.OFTModel(
        Path(args.weights).expanduser(),
        flip180=not args.no_flip,
        center_crop=not args.no_center_crop,
    )
    print(f"[smoke] load: {time.perf_counter() - t0:.1f}s")

    if torch.cuda.is_available():
        print(f"[smoke] VRAM allocated: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
        print(f"[smoke] VRAM reserved : {torch.cuda.memory_reserved() / 2**30:.2f} GiB")

    rng = np.random.default_rng(0)
    latencies = []
    for i in range(args.iters):
        obs = fake_obs(rng)
        t = time.perf_counter()
        chunk = model.predict_chunk(obs["image"], obs["wrist"], obs["state"], args.task)
        dt = time.perf_counter() - t
        latencies.append(dt)
        finite = np.isfinite(chunk).all()
        print(
            f"[smoke] {i}: {dt * 1000:7.1f} ms  shape={chunk.shape}"
            f"  min={chunk.min():+.3f} max={chunk.max():+.3f}"
            f"  finite={finite}"
        )
        if i == 0:
            print("[smoke] chunk[0] =", np.array2string(chunk[0], precision=4))

    if torch.cuda.is_available():
        print(f"[smoke] VRAM peak     : {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    warm = latencies[1:] or latencies
    print(
        f"[smoke] latency: first={latencies[0] * 1000:.0f} ms"
        f" warm_mean={np.mean(warm) * 1000:.0f} ms"
        f" warm_max={np.max(warm) * 1000:.0f} ms  (制限 10,000 ms)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
