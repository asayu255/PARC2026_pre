"""submission/oft_policy.py の単体テスト。

checkpoint 本体（15 GB）を読まずに確かめられることだけを見る。守りたいのは
ヘッドの形と画像・状態の前処理で、そこが違っていても OFT は例外を出さずに
それらしい action を返してしまう（π0.5 で黙ってランダム初期化のまま起動
しかけたのと同じ種類の事故）。

ヘッドの期待値は実物の state_dict をそのまま写したものである:

    action head (bf16)
      module.model.layer_norm1.{weight,bias}          (28672,)
      module.model.fc1.weight                         (4096, 28672)
      module.model.mlp_resnet_blocks.{0,1}.ffn.0.*    (4096,)
      module.model.mlp_resnet_blocks.{0,1}.ffn.1.weight (4096, 4096)
      module.model.layer_norm2.{weight,bias}          (4096,)
      module.model.fc2.weight                         (7, 4096)

    proprio projector (fp32)
      module.fc1.weight  (4096, 8)
      module.fc2.weight  (4096, 4096)
"""
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "submission"))

import oft_policy  # noqa: E402


# --- ヘッドの形 -------------------------------------------------------------


def test_action_head_matches_the_checkpoint_shapes():
    pytest.importorskip("torch")
    head_cls, _ = oft_policy._build_head_classes()
    shapes = {k: tuple(v.shape) for k, v in head_cls().state_dict().items()}

    assert shapes["model.layer_norm1.weight"] == (28672,)
    assert shapes["model.fc1.weight"] == (4096, 28672)
    assert shapes["model.mlp_resnet_blocks.0.ffn.0.weight"] == (4096,)
    assert shapes["model.mlp_resnet_blocks.0.ffn.1.weight"] == (4096, 4096)
    assert shapes["model.mlp_resnet_blocks.1.ffn.1.weight"] == (4096, 4096)
    assert shapes["model.layer_norm2.weight"] == (4096,)
    assert shapes["model.fc2.weight"] == (7, 4096)
    # ブロックはちょうど 2 つ。3 つあると checkpoint に無いキーが増える
    assert not any(k.startswith("model.mlp_resnet_blocks.2.") for k in shapes)


def test_proprio_projector_matches_the_checkpoint_shapes():
    pytest.importorskip("torch")
    _, proj_cls = oft_policy._build_head_classes()
    shapes = {k: tuple(v.shape) for k, v in proj_cls().state_dict().items()}

    assert shapes == {
        "fc1.weight": (4096, 8),
        "fc1.bias": (4096,),
        "fc2.weight": (4096, 4096),
        "fc2.bias": (4096,),
    }


def test_action_head_predicts_one_action_per_chunk_step():
    torch = pytest.importorskip("torch")
    head_cls, _ = oft_policy._build_head_classes()
    head = head_cls()
    hidden = torch.zeros(1, oft_policy.NUM_ACTIONS_CHUNK * oft_policy.ACTION_DIM, 4096)

    out = head.predict_action(hidden)

    assert out.shape == (1, oft_policy.NUM_ACTIONS_CHUNK, oft_policy.ACTION_DIM)


def test_module_prefix_from_ddp_is_stripped(tmp_path):
    torch = pytest.importorskip("torch")
    _, proj_cls = oft_policy._build_head_classes()
    saved = {f"module.{k}": v for k, v in proj_cls().state_dict().items()}
    path = tmp_path / "proprio_projector--1_checkpoint.pt"
    torch.save(saved, path)

    # strict=True で通ることが確認になる（余計なキーがあれば落ちる）
    oft_policy._load_checkpoint_into(proj_cls(), path, torch)


# --- 画像の前処理 -----------------------------------------------------------


def test_resize_lanczos_produces_224_uint8():
    pytest.importorskip("PIL")
    img = np.random.default_rng(0).integers(0, 256, (128, 128, 3), dtype=np.uint8)

    out = oft_policy._resize_lanczos(img, 224)

    assert out.shape == (224, 224, 3)
    assert out.dtype == np.uint8


def test_resize_is_a_noop_when_already_at_target():
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    assert oft_policy._resize_lanczos(img, 224) is img


def test_center_crop_at_scale_one_is_the_identity():
    """crop_scale=1.0 なら box は画像全体になり、align_corners=True の
    grid_sample は画素をそのまま拾う。ここがずれていれば 0.9 でもずれている。"""
    torch = pytest.importorskip("torch")
    img = np.random.default_rng(1).integers(0, 256, (32, 32, 3), dtype=np.uint8)

    out = oft_policy._center_crop(img, torch, crop_scale=1.0)

    assert np.array_equal(out, img)


def test_center_crop_zooms_in_by_the_square_root_of_the_scale():
    """中央だけ白い画像を拡大すると、白い領域が 1/sqrt(crop_scale) 倍になる。

    本番の 0.9 だと倍率が 1.054 で、16 px の正方形は 16.9 px にしかならず
    閾値の丸めに埋もれる。倍率が明確な 0.25（= 2 倍）で見る。
    """
    torch = pytest.importorskip("torch")
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[24:40, 24:40] = 255                      # 16 px 四方、中央

    out = oft_policy._center_crop(img, torch, crop_scale=0.25)

    assert out.shape == img.shape
    white_cols = (out[:, :, 0] > 127).any(axis=0).sum()
    assert 28 <= white_cols <= 32                # 16 px が概ね 2 倍


def test_center_crop_keeps_the_bright_region_centred():
    torch = pytest.importorskip("torch")
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[24:40, 24:40] = 255

    out = oft_policy._center_crop(img, torch, crop_scale=0.9)

    assert out.shape == img.shape
    rows = np.nonzero((out[:, :, 0] > 127).any(axis=1))[0]
    assert abs((rows[0] + rows[-1]) / 2 - 31.5) < 1.0


def test_flip180_is_applied_before_resize():
    pytest.importorskip("PIL")
    torch = pytest.importorskip("torch")
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[:8, :8] = 255                       # 左上だけ白

    out = oft_policy.prepare_image(img, torch, flip180=True, center_crop=False)

    assert out[:16, :16].mean() < out[-16:, -16:].mean()


# --- gripper の変換 ---------------------------------------------------------
#
# 本家 process_action は env.step の直前で
#   normalize_gripper_action(binarize=True) -> invert_gripper_action
# を通す。合成すると -sign(2g - 1) である。ここを落とすとグリッパーが
# 常に逆に動き、しかも例外は出ない。


def test_gripper_open_becomes_minus_one():
    """モデルは開くとき 1 付近を出す（統計は q99=1.0）。環境は -1 が開く。"""
    chunk = np.zeros((8, 7), dtype=np.float32)
    chunk[:, -1] = 1.04                       # 実測で出た値

    out = oft_policy.process_gripper(chunk)

    assert np.all(out[:, -1] == -1.0)


def test_gripper_closed_becomes_plus_one():
    chunk = np.zeros((8, 7), dtype=np.float32)
    chunk[:, -1] = 0.0                        # q01 側

    out = oft_policy.process_gripper(chunk)

    assert np.all(out[:, -1] == 1.0)


def test_gripper_without_binarize_is_the_linear_map():
    chunk = np.zeros((1, 7), dtype=np.float32)
    chunk[0, -1] = 0.25

    out = oft_policy.process_gripper(chunk, binarize=False)

    assert out[0, -1] == pytest.approx(-(2 * 0.25 - 1))   # = +0.5


def test_gripper_transform_leaves_the_other_dims_untouched():
    rng = np.random.default_rng(3)
    chunk = rng.normal(size=(8, 7)).astype(np.float32)

    out = oft_policy.process_gripper(chunk)

    assert np.array_equal(out[:, :6], chunk[:, :6])


def test_gripper_transform_does_not_mutate_its_input():
    chunk = np.ones((2, 7), dtype=np.float32)

    oft_policy.process_gripper(chunk)

    assert np.all(chunk[:, -1] == 1.0)


# --- 状態の正規化 -----------------------------------------------------------


def test_normalize_proprio_maps_the_quantiles_to_the_unit_interval():
    stats = {"q01": [-1.0, 0.0], "q99": [1.0, 10.0]}

    assert np.allclose(oft_policy.normalize_proprio(np.array([-1.0, 0.0]), stats), [-1, -1])
    assert np.allclose(oft_policy.normalize_proprio(np.array([1.0, 10.0]), stats), [1, 1])
    assert np.allclose(oft_policy.normalize_proprio(np.array([0.0, 5.0]), stats), [0, 0])


def test_normalize_proprio_clips_outside_the_quantiles():
    stats = {"q01": [-1.0], "q99": [1.0]}

    assert oft_policy.normalize_proprio(np.array([99.0]), stats)[0] == pytest.approx(1.0)
    assert oft_policy.normalize_proprio(np.array([-99.0]), stats)[0] == pytest.approx(-1.0)


def test_normalize_proprio_leaves_masked_out_dims_alone():
    stats = {"q01": [-1.0, -1.0], "q99": [1.0, 1.0], "mask": [True, False]}

    out = oft_policy.normalize_proprio(np.array([0.0, 7.0]), stats)

    # 分母の +1e-8（本家と同じ）で厳密な 0 にはならない
    assert out[0] == pytest.approx(0.0, abs=1e-6)
    assert out[1] == pytest.approx(7.0)


# --- checkpoint の判定 ------------------------------------------------------


def test_is_oft_checkpoint_reads_model_type(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "openvla"}))
    assert oft_policy.is_oft_checkpoint(tmp_path)


def test_is_oft_checkpoint_is_false_for_smolvla(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"type": "smolvla"}))
    assert not oft_policy.is_oft_checkpoint(tmp_path)


def test_is_oft_checkpoint_is_false_without_a_config(tmp_path):
    assert not oft_policy.is_oft_checkpoint(tmp_path)


# --- vendoring したシム -----------------------------------------------------


def test_the_prismatic_shim_exposes_what_modeling_prismatic_imports():
    """checkpoint 同梱の modeling_prismatic.py が import する名前がすべて有る。"""
    oft_policy._install_vendor_path()
    from prismatic.training.train_utils import (  # noqa: F401
        get_current_action_mask,
        get_next_actions_mask,
    )
    from prismatic.vla.constants import (  # noqa: F401
        ACTION_DIM,
        ACTION_PROPRIO_NORMALIZATION_TYPE,
        ACTION_TOKEN_BEGIN_IDX,
        IGNORE_INDEX,
        NUM_ACTIONS_CHUNK,
        STOP_INDEX,
        NormalizationType,
    )

    assert ACTION_DIM == 7
    assert NUM_ACTIONS_CHUNK == 8
    assert ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99
