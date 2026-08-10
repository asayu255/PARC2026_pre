#!/usr/bin/env python3
"""学習画像を評価解像度まで落としてから lerobot-train を走らせるラッパー。

    python tools/train_lora_lowres.py <lerobot-train と同じ引数...>

`tools/train_lora.sh` が `PARC_LORA_LOWRES` 付きで呼ぶので、通常は直接
叩かない。単体で使う場合は lerobot-train の引数をそのまま渡せばよい。

背景と何をするかは tools/lowres_transform.py の docstring を見ること。
要点だけ書くと、ベースの学習データは 256x256、PARC の評価は 128x128 で、
SmolVLA はどちらも 512 へ引き伸ばすので**差は鮮鋭度だけ**である。
その差を学習側で消す。

つまみ（環境変数）:
    PARC_LORA_LOWRES_RES=128     落とす先の解像度。評価と同じ 128 が既定
    PARC_LORA_LOWRES_MODE=down   down: 128 のまま渡す（評価と同一経路）
                                 roundtrip: 128 へ落として元サイズへ戻す

なぜ monkeypatch なのか
-----------------------
lerobot の `image_transforms` は augmentation 用の仕組みで、
`RandomSubsetApply` で確率的に一部だけ適用される。解像度の劣化は
**全フレームに決定的に**掛けたいので、その枠組みには乗らない。
また `--dataset.image_transforms.enable=false`（既定）のとき
`make_dataset` は `image_transforms=None` にしてしまう。

そこで `LeRobotDataset.__init__` の後ろで `self.image_transforms` を
差し替える。データセットは `__getitem__` の中で毎回 `self.image_transforms`
を読む（lerobot_dataset.py:1104）ので、後から差し替えても効く。
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lowres_transform import MODES, ComposeAfter, ResolutionDegradation  # noqa: E402


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[lowres] {name}={raw!r} を整数として読めない。既定 {default} を使う。")
        return default


def install_patch(res: int, mode: str) -> None:
    """LeRobotDataset が作られるたびに劣化変換を挿し込む。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    degrade = ResolutionDegradation(res=res, mode=mode)
    original_init = LeRobotDataset.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.image_transforms = ComposeAfter(self.image_transforms, degrade)
        # 1 回だけ、実際に効いていることを目に見える形で残す。
        # 学習ログを後から読んだときに、この run が低解像度だったか判る。
        if not getattr(LeRobotDataset, "_lowres_announced", False):
            LeRobotDataset._lowres_announced = True
            print(
                f"[lowres] 画像を {res}x{res} へ劣化させる（mode={mode}）。"
                f" カメラ: {getattr(self.meta, 'camera_keys', '?')}",
                flush=True,
            )

    LeRobotDataset.__init__ = patched_init


def main() -> None:
    res = _env_int("PARC_LORA_LOWRES_RES", 128)
    mode = os.environ.get("PARC_LORA_LOWRES_MODE", "down")
    if mode not in MODES:
        raise SystemExit(f"PARC_LORA_LOWRES_MODE は {MODES} のいずれか（{mode!r} が指定された）")

    print(f"[lowres] ラッパー経由で lerobot-train を起動する（res={res} mode={mode}）", flush=True)
    install_patch(res, mode)

    from lerobot.scripts.lerobot_train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
