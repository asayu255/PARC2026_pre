"""学習画像を評価時の解像度まで落とす劣化変換。

なぜ要るか
----------
ベースモデル（`lerobot/smolvla_libero_plus`）の学習データは 256x256 だが、
PARC の評価が渡す観測は **128x128** である（`pipeline/config.py:51` の
`LIBERO_EVAL_CAMERA` の既定値。評価側の設定なので提出物からは変えられない）。

SmolVLA は入ってきた画像を必ず 512x512 へ引き伸ばしてから使う
（`SmolVLAPolicy.prepare_images` -> `resize_with_pad`、
`config.resize_imgs_with_padding = (512, 512)`）。したがってモデルが見る
テンソルのサイズは学習時も評価時も 512 で同じで、**違うのは鮮鋭度だけ**である。

    学習: 256 --(bilinear 2 倍)--> 512
    評価: 128 --(bilinear 4 倍)--> 512   ← こちらのほうがぼける

§27 の LoRA 追加学習が 82.5% -> 50.0% に落ちた件で、5 仮説を潰したあとに
残った説明がこれである。追加学習は「評価時に存在しない鮮明さ」へモデルを
適合させていた可能性がある。この変換は学習側の画像を評価側と同じ鮮鋭度まで
落として、その差を消す。

何をするか
----------
`mode="down"`（既定）: 画像を `res` x `res` へ縮小して**そのまま返す**。
    モデルへの入力が評価時と同じ 128 になるので、そこから先の
    `resize_with_pad` は評価と完全に同一の経路を通る。

`mode="roundtrip"`: `res` へ縮小してから元のサイズへ戻す。
    情報量は down と同じだが補間が 1 段増えるので評価とは厳密には一致しない。
    画像サイズを変えたくない場合の逃げ道として残してある。

縮小には `antialias=True` を使う。評価側の 128x128 は MuJoCo が最初から
128 で描いたものなので、エイリアスの乗った素朴な間引きより、
帯域制限された縮小のほうが近い。

この変換は**オーグメンテーションではない**。全フレームに決定的に適用する。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

MODES = ("down", "roundtrip")


class ResolutionDegradation(torch.nn.Module):
    """画像を res へ縮小する（mode="roundtrip" なら元サイズへ戻す）。

    入力は lerobot のデータセットが返す形、すなわち float の (C, H, W) か、
    時間軸のある (T, C, H, W)。値域 [0, 1] を保つ。
    """

    def __init__(self, res: int = 128, mode: str = "down"):
        super().__init__()
        if res < 1:
            raise ValueError(f"res は 1 以上。受け取った値: {res}")
        if mode not in MODES:
            raise ValueError(f"mode は {MODES} のいずれか。受け取った値: {mode!r}")
        self.res = int(res)
        self.mode = mode

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if img.ndim < 3:
            raise ValueError(f"(..., C, H, W) を期待したが {tuple(img.shape)}")

        lead, (c, h, w) = img.shape[:-3], img.shape[-3:]
        if h == self.res and w == self.res:
            return img                      # 既に評価と同じ解像度。触らない

        flat = img.reshape(-1, c, h, w)
        # float でないと interpolate できない。uint8 で来ることは無いはずだが、
        # 来ても壊さずに戻す。
        orig_dtype = flat.dtype
        if not torch.is_floating_point(flat):
            flat = flat.float()

        low = F.interpolate(
            flat, size=(self.res, self.res),
            mode="bilinear", align_corners=False, antialias=True,
        )
        if self.mode == "roundtrip":
            low = F.interpolate(
                low, size=(h, w), mode="bilinear", align_corners=False,
            )

        out = low.to(orig_dtype).reshape(*lead, c, *low.shape[-2:])
        return out

    def __repr__(self) -> str:
        return f"{type(self).__name__}(res={self.res}, mode={self.mode!r})"


class ComposeAfter(torch.nn.Module):
    """既存の image_transforms（あれば）を先に通してから劣化をかける。

    オーグメンテーションは元の解像度で効かせたいので順序はこの向きに固定する。
    `--dataset.image_transforms.enable=true` を併用したときだけ意味を持つ。
    """

    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second

    def forward(self, x):
        return self.second(self.first(x) if self.first is not None else x)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.first!r} -> {self.second!r})"


def sharpness(img: torch.Tensor) -> float:
    """鮮鋭度の目安。隣接画素差の RMS。劣化が効いたかの確認に使う。"""
    x = img.reshape(-1, *img.shape[-3:]).float()
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return float(torch.sqrt((dx.pow(2).mean() + dy.pow(2).mean()) / 2))
