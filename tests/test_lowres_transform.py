"""tools/lowres_transform.py の単体テスト。

いちばん確かめたいのは **「学習時にモデルへ入るテンソルが、評価時と同じ
経路を通るか」** である。SmolVLA は入力を必ず 512 へ引き伸ばすので
（`resize_with_pad`）、劣化後の画像が評価時の 128x128 と同じ形・同じ値なら、
そこから先は同一の関数を通る = 完全に一致する。

lerobot が入っていれば `resize_with_pad` は本物を使い、無ければ下の写しを
使う。写しが本物と一致することも（lerobot がある環境では）検証する。
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "tools"))

from lowres_transform import ComposeAfter, ResolutionDegradation, sharpness  # noqa: E402


def resize_with_pad_ref(img, width, height, pad_value=-1):
    """lerobot 0.4.4 modeling_smolvla.py:135 の写し（動作の基準にする）。"""
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")
    cur_height, cur_width = img.shape[2:]
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )
    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))
    return F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)


def scene(size=256, seed=0):
    """細かい模様を含む合成画像。ぼかしの効果が測れるように高周波を入れる。"""
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(3, size, size, generator=g)
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing="ij"
    )
    stripes = ((xx * size / 2).floor() % 2) * 0.5      # 1 画素おきの縞
    return (base * 0.3 + stripes).clamp(0, 1)


# --- 形と値域 ---------------------------------------------------------------


def test_down_mode_returns_eval_resolution():
    out = ResolutionDegradation(res=128, mode="down")(scene(256))
    assert out.shape == (3, 128, 128)
    assert out.dtype == torch.float32
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0


def test_roundtrip_mode_preserves_size():
    out = ResolutionDegradation(res=128, mode="roundtrip")(scene(256))
    assert out.shape == (3, 256, 256)


def test_time_dimension_is_handled():
    out = ResolutionDegradation(res=128)(torch.stack([scene(256), scene(256, 1)]))
    assert out.shape == (2, 3, 128, 128)


def test_noop_when_already_at_target():
    x = scene(128)
    out = ResolutionDegradation(res=128)(x)
    assert out is x                       # 余計な補間を掛けない


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        ResolutionDegradation(res=0)
    with pytest.raises(ValueError):
        ResolutionDegradation(mode="blur")
    with pytest.raises(ValueError):
        ResolutionDegradation()(torch.zeros(4, 4))


# --- 劣化が実際に起きていること ----------------------------------------------


def test_degradation_removes_high_frequency():
    x = scene(256)
    rt = ResolutionDegradation(res=128, mode="roundtrip")(x)
    # 同じ 256x256 同士で比べる。縞が潰れるぶん鮮鋭度は明確に下がる。
    assert sharpness(rt) < sharpness(x) * 0.7


def test_is_deterministic():
    """augmentation ではない。同じ入力からは必ず同じ出力が出る。"""
    x, tf = scene(256), ResolutionDegradation(res=128)
    assert torch.equal(tf(x), tf(x))


# --- 本題: 評価時の経路と一致すること ----------------------------------------


def test_down_mode_matches_the_eval_path_exactly():
    """劣化後の画像は、評価が渡す 128x128 と同じ形式である。

    したがって `resize_with_pad` 以降はモデルにとって区別が付かない。
    ここでは「評価時の 128 画像」を、同じ帯域制限縮小で作ったものとして
    模擬し、512 まで通した結果が一致することを確かめる。
    """
    x256 = scene(256)
    degraded = ResolutionDegradation(res=128, mode="down")(x256)

    eval_like = F.interpolate(
        x256[None], size=(128, 128), mode="bilinear", align_corners=False, antialias=True
    )[0]

    assert torch.equal(degraded, eval_like)
    assert torch.equal(
        resize_with_pad_ref(degraded[None], 512, 512, pad_value=0),
        resize_with_pad_ref(eval_like[None], 512, 512, pad_value=0),
    )


def test_roundtrip_mode_does_not_match_the_eval_path():
    """roundtrip は補間が 1 段多いので厳密には一致しない（既定を down にした理由）。"""
    x256 = scene(256)
    rt = ResolutionDegradation(res=128, mode="roundtrip")(x256)
    down = ResolutionDegradation(res=128, mode="down")(x256)

    a = resize_with_pad_ref(rt[None], 512, 512, pad_value=0)
    b = resize_with_pad_ref(down[None], 512, 512, pad_value=0)
    assert not torch.allclose(a, b, atol=1e-3)
    # ただし情報量は同じなので、無劣化よりはずっと近い
    raw = resize_with_pad_ref(x256[None], 512, 512, pad_value=0)
    assert (a - b).abs().mean() < (raw - b).abs().mean()


def test_reference_resize_matches_lerobot():
    """写しが本物と一致すること（lerobot がある環境でのみ実行）。"""
    smolvla = pytest.importorskip("lerobot.policies.smolvla.modeling_smolvla")
    x = scene(128)[None]
    assert torch.equal(
        smolvla.resize_with_pad(x, 512, 512, pad_value=0),
        resize_with_pad_ref(x, 512, 512, pad_value=0),
    )


# --- 既存の transform との合成 ------------------------------------------------


def test_compose_applies_augmentation_first():
    calls = []

    class Marker(torch.nn.Module):
        def forward(self, x):
            calls.append(tuple(x.shape[-2:]))
            return x

    ComposeAfter(Marker(), ResolutionDegradation(res=128))(scene(256))
    assert calls == [(256, 256)]          # 拡張は元の解像度で掛かる


def test_compose_tolerates_no_existing_transform():
    out = ComposeAfter(None, ResolutionDegradation(res=128))(scene(256))
    assert out.shape == (3, 128, 128)


# --- ラッパーの monkeypatch ---------------------------------------------------
# 学習を実際に回さずに、patch が「データセットが読む場所」を書き換えることを
# 確かめる。lerobot は入っていなくてよいので偽のモジュールを立てる。


def _fake_lerobot(monkeypatch):
    """lerobot.datasets.lerobot_dataset.LeRobotDataset だけを持つ偽モジュール。"""
    import types

    class FakeDataset:
        def __init__(self, image_transforms=None):
            self.image_transforms = image_transforms
            self.meta = types.SimpleNamespace(camera_keys=["observation.images.front"])

        def getitem(self, img):
            # 本物の lerobot_dataset.py:1104 と同じで、毎回 self から読む
            if self.image_transforms is not None:
                return self.image_transforms(img)
            return img

    mod = types.ModuleType("lerobot.datasets.lerobot_dataset")
    mod.LeRobotDataset = FakeDataset
    pkg = types.ModuleType("lerobot")
    datasets_pkg = types.ModuleType("lerobot.datasets")
    monkeypatch.setitem(sys.modules, "lerobot", pkg)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", datasets_pkg)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", mod)
    return FakeDataset


def test_patch_degrades_even_when_transforms_are_disabled(monkeypatch):
    """enable=false だと lerobot は image_transforms=None にする。それでも効く。"""
    FakeDataset = _fake_lerobot(monkeypatch)
    import train_lora_lowres

    train_lora_lowres.install_patch(res=128, mode="down")

    ds = FakeDataset(image_transforms=None)
    assert ds.getitem(scene(256)).shape == (3, 128, 128)


def test_patch_keeps_existing_augmentation(monkeypatch):
    FakeDataset = _fake_lerobot(monkeypatch)
    import train_lora_lowres

    train_lora_lowres.install_patch(res=128, mode="down")

    seen = []

    def aug(x):
        seen.append(tuple(x.shape[-2:]))
        return x

    ds = FakeDataset(image_transforms=aug)
    out = ds.getitem(scene(256))

    assert seen == [(256, 256)]           # 拡張が消えていない、かつ先に掛かる
    assert out.shape == (3, 128, 128)
