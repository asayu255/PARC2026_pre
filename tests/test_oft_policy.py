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


# --- 提出物に同梱する構成ファイル -------------------------------------------
#
# 採点環境では環境変数を渡せないので、提出ごとの構成は parc_env が持つ。
# ここが壊れると「A/B のつもりで同じものを 2 回提出する」事故になり、
# しかも結果を見ても区別がつかない。


def _exec_loader(tmp_path):
    """policy_server.py から _load_submission_env だけを取り出して動かす。

    モジュールごと import すると torch や重みのロードまで走るので、対象の
    関数だけを切り出す。終端は行頭の呼び出し `\n_load_submission_env()` で
    探す。`_load_submission_env()` をそのまま探すと **定義行そのもの**
    （`def _load_submission_env() -> None:`）にヒットして 4 文字しか
    取り出せない。
    """
    import os

    src = (_ROOT / "submission" / "policy_server.py").read_text()
    start = src.index("def _load_submission_env")
    end = src.index("\n_load_submission_env()", start)
    ns = {"os": os, "_HERE": tmp_path}
    exec(src[start:end], ns)          # noqa: S102 - 関数 1 つだけを取り出して動かす
    assert "_load_submission_env" in ns, "関数を取り出せていない"
    return ns


def _load_env_from(tmp_path, text, monkeypatch, preset=None):
    for k, v in (preset or {}).items():
        monkeypatch.setenv(k, v)
    (tmp_path / "parc_env").write_text(text)

    ns = _exec_loader(tmp_path)
    ns["_load_submission_env"]()


def test_parc_env_sets_the_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("PARC_ENSEMBLE", raising=False)
    _load_env_from(tmp_path, "PARC_ENSEMBLE=1\nPARC_ENS_H=8\n", monkeypatch)

    import os

    assert os.environ["PARC_ENSEMBLE"] == "1"
    assert os.environ["PARC_ENS_H"] == "8"


def test_a_real_environment_variable_wins(tmp_path, monkeypatch):
    """スイープが渡した値を提出物のファイルが握り潰してはいけない。"""
    _load_env_from(
        tmp_path, "PARC_ENSEMBLE=1\n", monkeypatch, preset={"PARC_ENSEMBLE": "0"}
    )

    import os

    assert os.environ["PARC_ENSEMBLE"] == "0"


def test_comments_and_blank_lines_are_skipped(tmp_path, monkeypatch):
    monkeypatch.delenv("PARC_ENS_M", raising=False)
    _load_env_from(tmp_path, "\n# 説明\nPARC_ENS_M=0.01  # 末尾コメント\n", monkeypatch)

    import os

    assert os.environ["PARC_ENS_M"] == "0.01"


def test_keys_outside_the_parc_namespace_are_ignored(tmp_path, monkeypatch):
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    _load_env_from(tmp_path, "LD_PRELOAD=/evil.so\n", monkeypatch)

    import os

    assert "LD_PRELOAD" not in os.environ


def test_a_missing_file_is_not_an_error(tmp_path):
    _exec_loader(tmp_path)["_load_submission_env"]()      # 例外が出ないこと


# --- TTA（空間方向の平均）-----------------------------------------------------
#
# temporal ensembling は時間方向の平均で、L1 回帰ヘッドの OFT は出力が決定的
# なので、同じ画像を何度推論しても同じ値しか出ない。入力側を振る必要がある。


def test_tta_crop_scales_are_distinct_and_centred_on_the_default():
    scales = oft_policy.TTA_CROP_SCALES

    assert scales[0] == oft_policy.CENTER_CROP_SCALE   # 1 視点なら既定と同じ
    assert len(set(scales)) == len(scales)
    assert all(0.5 < x <= 1.0 for x in scales)         # 分布内に留める


def test_every_prefix_of_the_crop_scales_stays_centred():
    """PARC_OFT_TTA=N は先頭 N 個を使う。どの N でも平均が既定の近くに居ること。

    片側に寄った集合で平均すると、視点を増やすほど系統的にズームが偏る。
    TTA は分散を減らすためのものなので、偏りを持ち込んでは意味が無い。
    偶数個では必ず片側が 1 つ多くなるため、厳密な対称は要求しない。
    """
    scales = oft_policy.TTA_CROP_SCALES
    default = oft_policy.CENTER_CROP_SCALE

    for n in range(1, len(scales) + 1):
        mean = sum(scales[:n]) / n
        assert abs(mean - default) <= 0.03, f"{n} 視点の平均が {mean:.4f} で偏っている"


def test_the_shipped_prefix_is_frozen():
    """先頭 2 つは tta2 として採点で 0.304 を出した構成である。

    ここを並べ替えると、いま提出している設定が黙って別物になる。
    """
    assert oft_policy.TTA_CROP_SCALES[:2] == (0.90, 0.95)


def test_no_view_skips_the_crop():
    """倍率 1.00 のクロップは恒等変換で、「クロップしない」条件と同値である。

    それは単独で測って悪化した条件（`PARC_OFT_CENTER_CROP=0`）なので、
    平均に混ぜてはいけない。旧版は 4 番目が 1.00 で、その `tta4` は採点
    0.271（`tta2` は 0.304）だった。
    """
    assert all(x < 1.0 for x in oft_policy.TTA_CROP_SCALES)


def test_the_first_bank_is_the_shipped_one():
    """0 番目の予備列は既定と同一でなければならない。

    ここがずれると、脱出が一度も起きなかったエピソードの挙動まで変わる。
    """
    assert oft_policy.TTA_CROP_BANKS[0] is oft_policy.TTA_CROP_SCALES


def test_every_bank_is_balanced_around_the_nominal_crop():
    """どの列も、先頭 N 個の平均が公称 0.90 でなければならない（N は奇数）。

    `tta2` は平均が 0.925 と有害側へずれて ep6 を 119 -> 214 step にした。
    脱出で列を差し替えるときに同じ罠を踏まないための固定である。
    """
    for i, bank in enumerate(oft_policy.TTA_CROP_BANKS):
        for n in (1, 3, 5, 7):
            assert sum(bank[:n]) / n == pytest.approx(0.90), (i, n)


def test_every_bank_avoids_the_identity_crop():
    for bank in oft_policy.TTA_CROP_BANKS:
        assert all(x < 1.0 for x in bank)


def test_the_banks_actually_differ_at_the_shipped_view_count():
    """脱出は「同じ状態から別の行動」が出ないと意味が無い。"""
    seen = {bank[:3] for bank in oft_policy.TTA_CROP_BANKS}
    assert len(seen) == len(oft_policy.TTA_CROP_BANKS)


def _bare_model(scales, views, pinned=False):
    """__init__ を通さずに倍率列まわりだけ組み立てる（重みを読まない）。"""
    m = oft_policy.OFTModel.__new__(oft_policy.OFTModel)
    m.crop_scales = scales
    m.tta_views = views
    m._bank = 0
    m._scales_pinned = pinned
    return m


def test_rotation_walks_the_banks_and_then_stops():
    m = _bare_model(oft_policy.TTA_CROP_SCALES, 3)
    first = m.rotate_crop_scales()
    second = m.rotate_crop_scales()

    assert first == oft_policy.TTA_CROP_BANKS[1][:3]
    assert second == oft_policy.TTA_CROP_BANKS[2][:3]
    assert m.rotate_crop_scales() is None       # 予備は 2 本きり


def test_rotation_is_a_noop_for_a_single_view():
    """1 視点では全列の先頭が 0.90 で同一。黙って空振りしない。"""
    assert _bare_model(oft_policy.TTA_CROP_SCALES, 1).rotate_crop_scales() is None


def test_rotation_respects_an_explicit_scale_list():
    """PARC_OFT_TTA_SCALES で固定した指定を黙って破らない。"""
    m = _bare_model((0.9, 0.94, 0.86), 3, pinned=True)

    assert m.rotate_crop_scales() is None
    m.reset_crop_scales()
    assert m.crop_scales == (0.9, 0.94, 0.86)


def test_reset_puts_the_default_bank_back():
    m = _bare_model(oft_policy.TTA_CROP_SCALES, 3)
    m.rotate_crop_scales()

    m.reset_crop_scales()

    assert m.crop_scales is oft_policy.TTA_CROP_SCALES and m._bank == 0


def test_prepare_image_honours_an_explicit_crop_scale():
    pytest.importorskip("PIL")
    torch = pytest.importorskip("torch")
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[56:72, 56:72] = 255

    wide = oft_policy.prepare_image(img, torch, False, True, crop_scale=1.0)
    tight = oft_policy.prepare_image(img, torch, False, True, crop_scale=0.25)

    # きつく切るほど白い領域は大きく写る
    assert (tight > 127).sum() > (wide > 127).sum()


def test_prepare_image_without_crop_ignores_the_scale():
    pytest.importorskip("PIL")
    torch = pytest.importorskip("torch")
    img = np.random.default_rng(4).integers(0, 256, (128, 128, 3), dtype=np.uint8)

    a = oft_policy.prepare_image(img, torch, False, False, crop_scale=0.5)
    b = oft_policy.prepare_image(img, torch, False, False)

    assert np.array_equal(a, b)


# --- 倍率列の環境変数上書き -------------------------------------------------
#
# 採点が決定的だと分かったので、採点そのものを測定器として A/B する。倍率列を
# 変えるたびに 12 GB の zip を作り直さずに済むよう、環境変数で差し替えられる。


def test_crop_scales_default_to_the_frozen_tuple(monkeypatch):
    monkeypatch.delenv("PARC_OFT_TTA_SCALES", raising=False)
    assert oft_policy.resolve_crop_scales() == oft_policy.TTA_CROP_SCALES


def test_crop_scales_can_be_replaced(monkeypatch):
    monkeypatch.setenv("PARC_OFT_TTA_SCALES", "0.90,0.95,0.85,0.925")
    assert oft_policy.resolve_crop_scales() == (0.90, 0.95, 0.85, 0.925)


def test_crop_scales_tolerate_spaces_and_trailing_comma(monkeypatch):
    monkeypatch.setenv("PARC_OFT_TTA_SCALES", " 0.90 , 0.88 , ")
    assert oft_policy.resolve_crop_scales() == (0.90, 0.88)


@pytest.mark.parametrize("spec", ["", "   "])
def test_blank_falls_back_to_the_default(spec):
    assert oft_policy.resolve_crop_scales(spec) == oft_policy.TTA_CROP_SCALES


@pytest.mark.parametrize("spec", ["abc", "0.9,x", "1.00", "0.9,1.0", "0.3", "0,0.9"])
def test_bad_or_out_of_range_specs_fall_back(spec):
    """特に 1.00 を弾く。恒等クロップ = 単独で悪化が実測された条件である。

    綴りを間違えたまま静かに走るより、既定へ戻したほうが損害が小さい
    （採点は 1 回 18 分で、間違った構成を測ると 1 回まるごと無駄になる）。
    """
    assert oft_policy.resolve_crop_scales(spec) == oft_policy.TTA_CROP_SCALES


def test_the_override_is_what_the_model_would_use(monkeypatch):
    """resolve_crop_scales() の戻りがそのまま TTA の視点列になる。

    OFTModel はモデルを読むので組み立てられない。__init__ が計算する式と
    同じものをここで固定する。
    """
    monkeypatch.setenv("PARC_OFT_TTA_SCALES", "0.90,0.95,0.85")
    scales = oft_policy.resolve_crop_scales()

    for views, expected in [(1, (0.90,)), (3, (0.90, 0.95, 0.85)),
                            (8, (0.90, 0.95, 0.85))]:   # 列より多くは取れない
        n = max(1, min(views, len(scales)))
        assert scales[:n] == expected
