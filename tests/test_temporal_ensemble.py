"""submission/policy_server.py の temporal ensembling の単体テスト。

モデルは要らない。`_predict_chunk()` を「呼ばれた回数がわかる既知のチャンク」
に差し替えて、重み・重なる本数・エピソード境界・既定値を固定する。

ここで守りたいのは主に 2 つである。

- **既定が off のままであること。** 3 回目の採点 0.077 はこの経路で出した
  数字であり、A/B で確認するまで既定を動かしてはいけない
- ensembling を有効にしたときに、前エピソードの予測が次エピソードへ
  漏れないこと。漏れると衝突（1mm 判定）の直接の原因になる

MyPolicy は import 時に torch も lerobot も要求しないが、モジュール自体は
msgpack / fastapi / uvicorn を import する。提出側の依存なので、無ければ skip。
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "submission" / "policy_server.py"

for _mod in ("msgpack", "fastapi", "uvicorn"):
    if importlib.util.find_spec(_mod) is None:
        pytest.skip(f"{_mod} が無い（提出側の依存）", allow_module_level=True)

OBS = {
    "agentview_image": np.zeros((128, 128, 3), np.uint8),
    "robot0_eye_in_hand_image": np.zeros((128, 128, 3), np.uint8),
    "robot0_joint_pos": np.zeros(7),
    "robot0_eef_pos": np.zeros(3),
    "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
    "robot0_gripper_qpos": np.zeros(2),
}


def load_module(monkeypatch, **env):
    """PARC_* を差し替えた状態で policy_server.py を読み込み直す。

    つまみはクラス属性として import 時に確定するので、条件ごとに
    モジュールごと読み直す必要がある。
    """
    for key in [k for k in sys.modules if k.startswith("_ps_")]:
        del sys.modules[key]
    monkeypatch.delenv("PARC_ENSEMBLE", raising=False)
    for name in ("PARC_ENS_H", "PARC_ENS_QUERY", "PARC_ENS_M", "PARC_ENS_GRIPPER",
                 "PARC_N_EXEC", "PARC_DEBUG_DIR"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, str(value))

    name = f"_ps_{len(sys.modules)}"
    spec = importlib.util.spec_from_file_location(name, _SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_policy(mod, chunk_fn):
    """MyPolicy を __init__ を通さずに組み立てる（モデルロードと warmup を避ける）。"""
    p = mod.MyPolicy.__new__(mod.MyPolicy)
    p.instruction = ""
    p._queue = mod.deque()
    p._ens = mod.deque()
    p._step = 0
    p.torch = p.device = p.preprocessor = p.postprocessor = None
    p.state_dim = 8
    p.key_main, p.key_wrist = p.KEY_MAIN, p.KEY_WRIST
    p._dbg_n = p._trace_i = 0
    p._warming = True          # レイテンシ集計とデバッグダンプを黙らせる
    p._lat_max = p._lat_sum = 0.0
    p._lat_n = p._lat_slow = 0
    p.model = None
    p.calls = 0

    def _predict(obs):
        p.calls += 1
        return chunk_fn(p.calls - 1)

    p._predict_chunk = _predict
    return p


def ramp(step=0.01, n=50):
    """i 回目の推論が定数 step*(i+1) を返すチャンク。何回目の予測かが値で分かる。

    _sanitize() が [-1, 1] にクリップするので、値はその内側に収める。
    """
    return lambda i: np.full((n, 7), step * (i + 1), np.float32)


def wavg(values, m):
    """古い順に並んだ values に ACT の重み exp(-m*i) をかけた加重平均。"""
    v = np.asarray(values, float)
    w = np.exp(-m * np.arange(len(v)))
    return float((v * (w / w.sum())).sum())


def drive(policy, steps):
    return [float(policy.get_action(OBS)[0]) for _ in range(steps)]


# --- 既定値 -----------------------------------------------------------------


def test_defaults_keep_the_graded_configuration(monkeypatch):
    """既定は ensembling 無効・n_exec=5。採点 0.077 を出した構成である。"""
    cfg = load_module(monkeypatch).MyPolicy
    assert cfg.TEMPORAL_ENSEMBLE is False
    assert cfg.N_ACTION_EXEC == 5


def test_ensemble_defaults(monkeypatch):
    cfg = load_module(monkeypatch, PARC_ENSEMBLE=1).MyPolicy
    assert cfg.TEMPORAL_ENSEMBLE is True
    assert (cfg.ENSEMBLE_HORIZON, cfg.ENSEMBLE_QUERY_EVERY) == (16, 1)
    assert cfg.ENSEMBLE_M == 0.01          # ACT の k と同値
    assert cfg.ENSEMBLE_GRIPPER is True


# --- 無効時: 既存の n_exec 経路が変わっていないこと --------------------------


def test_disabled_path_is_unchanged(monkeypatch):
    mod = load_module(monkeypatch)
    p = make_policy(mod, ramp())

    got = drive(p, 11)

    assert p.calls == 3                                  # 5 ステップに 1 回
    assert got == pytest.approx([0.01] * 5 + [0.02] * 5 + [0.03])
    assert len(p._ens) == 0                              # バッファに触らない


# --- 有効時: 重なった予測の加重平均になること --------------------------------


def test_queries_every_step_and_averages_overlapping_predictions(monkeypatch):
    p = make_policy(load_module(monkeypatch, PARC_ENSEMBLE=1), ramp())

    got = drive(p, 30)

    assert p.calls == 30
    # 1 本しかない初手は平均のしようがない
    assert got[0] == pytest.approx(0.01)
    # step 3 では 1..4 回目の予測が重なる
    assert got[3] == pytest.approx(wavg([0.01, 0.02, 0.03, 0.04], 0.01))
    # step 20 では h=16 本で頭打ち。step j の推論の値は 0.01*(j+1)
    assert got[20] == pytest.approx(wavg([0.01 * (j + 1) for j in range(5, 21)], 0.01))


def test_constant_predictions_pass_through_unchanged(monkeypatch):
    """平均が値そのものを歪めない（バイアスを入れない）こと。"""
    mod = load_module(monkeypatch, PARC_ENSEMBLE=1)
    p = make_policy(mod, lambda i: np.full((50, 7), 0.25, np.float32))

    assert drive(p, 30) == pytest.approx([0.25] * 30)


def test_smooths_resampling_noise(monkeypatch):
    """狙いそのものの確認: 推論ごとのノイズが jerk を作り、平均がそれを消す。

    flow-matching は推論のたびにノイズを引き直すので、チャンク境界で前後の
    予測が食い違う。真値は滑らかなランプ、チャンクごとに独立なガウスノイズ、
    という模型で jerk を比べる。
    """
    def noisy():
        rng = np.random.default_rng(0)

        def f(i):
            base = np.linspace(0, 0.5, 50, dtype=np.float32)[:, None]
            return (base + rng.normal(0, 0.05, (50, 7))).astype(np.float32)
        return f

    def jerk(seq):
        return float(np.sqrt((np.diff(np.asarray(seq), 3, axis=0) ** 2).mean()))

    off = make_policy(load_module(monkeypatch), noisy())
    on = make_policy(load_module(monkeypatch, PARC_ENSEMBLE=1), noisy())

    j_off = jerk([off.get_action(OBS) for _ in range(150)])
    j_on = jerk([on.get_action(OBS) for _ in range(150)])

    assert j_on < j_off / 2


# --- つまみ -----------------------------------------------------------------


def test_query_every_reduces_inference_count(monkeypatch):
    mod = load_module(monkeypatch, PARC_ENSEMBLE=1, PARC_ENS_QUERY=4, PARC_ENS_H=16)
    p = make_policy(mod, ramp())

    got = drive(p, 40)

    assert p.calls == 10                                 # 4 ステップに 1 回
    # step 36 では step 24/28/32/36 の推論（7..10 回目）が重なる
    assert got[36] == pytest.approx(wavg([0.07, 0.08, 0.09, 0.10], 0.01))


def test_no_gap_when_predictions_run_out(monkeypatch):
    """query > h でも穴を作らず、尽きた時点で再推論する。"""
    mod = load_module(monkeypatch, PARC_ENSEMBLE=1, PARC_ENS_QUERY=8, PARC_ENS_H=3)
    p = make_policy(mod, lambda i: np.full((50, 7), 0.3, np.float32))

    assert drive(p, 40) == pytest.approx([0.3] * 40)
    assert p.calls > 40 // 8


@pytest.mark.parametrize("m, expected", [
    (0.5, "older"),      # ACT の向き（k=0.01 と同符号）
    (0.0, "uniform"),
    (-0.5, "newer"),     # 差分 action では古い予測が陳腐化するので、この向きも候補
])
def test_sign_of_m_selects_which_predictions_dominate(monkeypatch, m, expected):
    mod = load_module(monkeypatch, PARC_ENSEMBLE=1, PARC_ENS_M=m, PARC_ENS_H=16)
    p = make_policy(mod, ramp())

    got = drive(p, 3)[-1]        # 重なる値は古い順に 0.01, 0.02, 0.03

    if expected == "older":
        assert got < 0.02
    elif expected == "newer":
        assert got > 0.02
    else:
        assert got == pytest.approx(0.02)


def test_gripper_can_opt_out_of_averaging(monkeypatch):
    """PARC_ENS_GRIPPER=0 のとき action[6] だけ最新の予測をそのまま使う。

    LIBERO の gripper は実質 2 値なので、平均すると開閉が数ステップ鈍る。
    """
    mod = load_module(monkeypatch, PARC_ENSEMBLE=1, PARC_ENS_GRIPPER=0)
    p = make_policy(mod, ramp())

    for _ in range(3):
        action = p.get_action(OBS)

    assert action[6] == pytest.approx(0.03)      # 最新
    assert action[0] != pytest.approx(0.03)      # 他は平均のまま


def test_gripper_is_averaged_by_default(monkeypatch):
    mod = load_module(monkeypatch, PARC_ENSEMBLE=1)
    p = make_policy(mod, ramp())

    for _ in range(3):
        action = p.get_action(OBS)

    assert action[6] == pytest.approx(action[0])


# --- エピソード境界と防御 ----------------------------------------------------


def test_reset_drops_the_previous_episode(monkeypatch):
    """持ち越すと前エピソードの action が次の冒頭に流れ込み、衝突の原因になる。"""
    mod = load_module(monkeypatch, PARC_ENSEMBLE=1)
    p = make_policy(mod, ramp())
    drive(p, 10)

    p.reset("next task")

    assert len(p._ens) == 0
    assert p._step == 0
    assert p.instruction == "next task"
    # 11 回目の推論の値がそのまま出る = 前の 10 本が混ざっていない
    assert p.get_action(OBS)[0] == pytest.approx(0.11)


def test_bad_env_values_fall_back_to_defaults(monkeypatch):
    """つまみが壊れていてもサーバーは起動する（起動しないほうが損害が大きい）。"""
    cfg = load_module(monkeypatch, PARC_ENSEMBLE=1, PARC_ENS_M="abc",
                      PARC_ENS_H="0", PARC_ENS_QUERY="-3").MyPolicy

    assert cfg.ENSEMBLE_M == 0.01
    assert cfg.ENSEMBLE_HORIZON == 1
    assert cfg.ENSEMBLE_QUERY_EVERY == 1


def test_sanitize_still_guards_the_ensembled_output(monkeypatch):
    """評価側へ返す前の最終防衛は ensembling 経路でも効く。"""
    mod = load_module(monkeypatch, PARC_ENSEMBLE=1)

    p = make_policy(mod, lambda i: np.full((50, 7), 5.0, np.float32))
    action = p.get_action(OBS)
    assert action.shape == (7,) and action.dtype == np.float32
    assert action.max() <= 1.0

    p = make_policy(mod, lambda i: np.full((50, 7), np.nan, np.float32))
    assert not np.isnan(p.get_action(OBS)).any()


def test_empty_chunk_is_rejected(monkeypatch):
    mod = load_module(monkeypatch, PARC_ENSEMBLE=1)
    p = make_policy(mod, lambda i: np.zeros((0, 7), np.float32))

    with pytest.raises(RuntimeError, match="空の action"):
        p.get_action(OBS)
