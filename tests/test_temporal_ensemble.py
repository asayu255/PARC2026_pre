"""submission/policy_server.py の temporal ensembling の単体テスト。

モデルは要らない。`_predict_chunk()` を「呼ばれた回数がわかる既知のチャンク」
に差し替えて、重み・重なる本数・エピソード境界・既定値を固定する。

ここで守りたいのは主に 2 つである。

- **既定が on で h=8 であること。** stove 50 エピソードと公開 4 タスクの
  2 ラウンドで jerk が −38〜42% 再現したうえでこの既定にした（§29）。
  意図せず戻ると、その改善が黙って消える
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
                 "PARC_N_EXEC", "PARC_DEBUG_DIR",
                 "PARC_ACT_SCALE", "PARC_ACT_SCALE_ROT",
                 "PARC_ACT_SLEW", "PARC_ACT_SLEW_ROT", "PARC_GRIP_RETRY",
                 "PARC_GRIP_RETRY_AFTER", "PARC_GRIP_RETRY_OPEN",
                 "PARC_GRIP_RETRY_MAX", "PARC_GRIP_RETRY_MIN_STEP"):
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
    p._prev_action = None
    p.torch = p.device = p.preprocessor = p.postprocessor = None
    p.state_dim = 8
    p.key_main, p.key_wrist = p.KEY_MAIN, p.KEY_WRIST
    p._dbg_n = p._trace_i = 0
    p._warming = True          # レイテンシ集計とデバッグダンプを黙らせる
    p._lat_max = p._lat_sum = 0.0
    p._lat_n = p._lat_slow = 0
    p._d_xyz = []
    p._d_rot = []
    p._d_clip = 0
    p._ep_step = 0
    p._grip_seen = []
    p._grip_close_run = 0
    p._grip_open_left = 0
    p._grip_retries = 0
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


def test_defaults_are_the_adopted_configuration(monkeypatch):
    """既定は ensembling 有効・h=8。§29.9 で採用した構成である。"""
    cfg = load_module(monkeypatch).MyPolicy
    assert cfg.TEMPORAL_ENSEMBLE is True
    assert cfg.ENSEMBLE_HORIZON == 8
    assert cfg.ENSEMBLE_QUERY_EVERY == 1   # 毎ステップ推論
    assert cfg.ENSEMBLE_M == 0.01          # ACT の k と同値
    assert cfg.ENSEMBLE_GRIPPER is True


def test_ensembling_can_be_turned_off(monkeypatch):
    """PARC_ENSEMBLE=0 で従来の N_ACTION_EXEC 経路へ戻せる。"""
    cfg = load_module(monkeypatch, PARC_ENSEMBLE=0).MyPolicy
    assert cfg.TEMPORAL_ENSEMBLE is False
    assert cfg.N_ACTION_EXEC == 5


# --- 無効時: 既存の n_exec 経路が変わっていないこと --------------------------


def test_disabled_path_is_unchanged(monkeypatch):
    mod = load_module(monkeypatch, PARC_ENSEMBLE=0)
    p = make_policy(mod, ramp())

    got = drive(p, 11)

    assert p.calls == 3                                  # 5 ステップに 1 回
    assert got == pytest.approx([0.01] * 5 + [0.02] * 5 + [0.03])
    assert len(p._ens) == 0                              # バッファに触らない


# --- 有効時: 重なった予測の加重平均になること --------------------------------


def test_queries_every_step_and_averages_overlapping_predictions(monkeypatch):
    # h を明示する。既定は 8 だが、頭打ちの検証には広いほうが見やすい。
    p = make_policy(load_module(monkeypatch, PARC_ENS_H=16), ramp())

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

    off = make_policy(load_module(monkeypatch, PARC_ENSEMBLE=0), noisy())
    on = make_policy(load_module(monkeypatch), noisy())

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


# --- slew rate 制限 ---------------------------------------------------------
#
# ACT_SCALE との違いを守るためのテスト群である。ACT_SCALE は定常速度を落とす
# ので「掴めない」失敗を生んだ（公開 4 タスク中 3 つが 0）。slew 制限は跳ねだけ
# 削り、定常速度は保つ。その性質そのものを固定する。


def step_chunks(values):
    """i 回目の推論が定数 values[i] を返すチャンク。跳ねを作るのに使う。"""
    return lambda i: np.full((50, 7), values[min(i, len(values) - 1)], np.float32)


def test_slew_is_off_by_default(monkeypatch):
    """未測定のつまみを既定で入れない。0.304 の構成を黙って変えないこと。"""
    cfg = load_module(monkeypatch).MyPolicy

    assert cfg.ACT_SLEW == 0.0
    assert cfg.ACT_SLEW_ROT == 0.0


def test_slew_caps_the_jump_between_steps(monkeypatch):
    """0 -> 0.8 の跳ねが 0.1 刻みに均される。"""
    mod = load_module(monkeypatch, PARC_ENSEMBLE=0, PARC_N_EXEC=1, PARC_ACT_SLEW=0.1)
    p = make_policy(mod, step_chunks([0.0, 0.8, 0.8, 0.8, 0.8]))

    out = drive(p, 5)
    assert out[0] == pytest.approx(0.0)          # 先頭は基準が無いので素通し
    assert out[1] == pytest.approx(0.1)
    assert out[2] == pytest.approx(0.2)
    assert out[3] == pytest.approx(0.3)


def test_slew_does_not_slow_a_steady_command(monkeypatch):
    """定常速度は落とさない。ACT_SCALE との決定的な違い。"""
    mod = load_module(monkeypatch, PARC_ENSEMBLE=0, PARC_N_EXEC=1, PARC_ACT_SLEW=0.1)
    p = make_policy(mod, lambda i: np.full((50, 7), 0.05, np.float32))

    assert drive(p, 6) == pytest.approx([0.05] * 6)


def test_slew_reaches_the_commanded_value_and_stays(monkeypatch):
    """頭打ちは過渡だけ。数 step 後には指令値に追いつく（振幅は失わない）。"""
    mod = load_module(monkeypatch, PARC_ENSEMBLE=0, PARC_N_EXEC=1, PARC_ACT_SLEW=0.1)
    p = make_policy(mod, step_chunks([0.0, 0.35]))

    out = drive(p, 8)
    assert out[4] == pytest.approx(0.35)
    assert out[-1] == pytest.approx(0.35)


def test_slew_leaves_the_gripper_alone(monkeypatch):
    """開閉は二値。遅らせると掴み損なう。"""
    mod = load_module(monkeypatch, PARC_ENSEMBLE=0, PARC_N_EXEC=1, PARC_ACT_SLEW=0.05)
    p = make_policy(mod, step_chunks([-1.0, 1.0]))

    p.get_action(OBS)
    a = p.get_action(OBS)
    assert a[6] == pytest.approx(1.0)            # gripper は一気に反転する
    assert a[0] == pytest.approx(-0.95)          # 並進は 0.05 しか動かない


def test_slew_rot_can_differ_from_translation(monkeypatch):
    mod = load_module(monkeypatch, PARC_ENSEMBLE=0, PARC_N_EXEC=1,
                      PARC_ACT_SLEW=0.1, PARC_ACT_SLEW_ROT=0.02)
    p = make_policy(mod, step_chunks([0.0, 0.5]))

    p.get_action(OBS)
    a = p.get_action(OBS)
    assert a[:3] == pytest.approx([0.1] * 3)
    assert a[3:6] == pytest.approx([0.02] * 3)


def test_slew_rot_alone_leaves_translation_free(monkeypatch):
    """PARC_ACT_SLEW=0 でも回転だけ制限できる。"""
    mod = load_module(monkeypatch, PARC_ENSEMBLE=0, PARC_N_EXEC=1, PARC_ACT_SLEW_ROT=0.02)
    p = make_policy(mod, step_chunks([0.0, 0.5]))

    p.get_action(OBS)
    a = p.get_action(OBS)
    assert a[:3] == pytest.approx([0.5] * 3)
    assert a[3:6] == pytest.approx([0.02] * 3)


def test_slew_does_not_carry_across_episodes(monkeypatch):
    """前エピソード末尾の指令が次の先頭を縛らない（reset で基準を捨てる）。"""
    mod = load_module(monkeypatch, PARC_ENSEMBLE=0, PARC_N_EXEC=1, PARC_ACT_SLEW=0.1)
    p = make_policy(mod, step_chunks([-0.9, 0.9]))

    drive(p, 3)
    p.reset("next")
    assert p._prev_action is None
    assert p.get_action(OBS)[0] == pytest.approx(0.9)


def test_slew_output_stays_in_range(monkeypatch):
    """_sanitize の後に掛かるが、範囲は壊さない。"""
    mod = load_module(monkeypatch, PARC_ENSEMBLE=1, PARC_ACT_SLEW=0.1)
    p = make_policy(mod, lambda i: np.full((50, 7), 5.0, np.float32))

    for _ in range(5):
        a = p.get_action(OBS)
        assert a.shape == (7,) and a.dtype == np.float32
        assert a.min() >= -1.0 and a.max() <= 1.0


def test_delta_tracking_measures_the_raw_jump(monkeypatch):
    """上限を選ぶための計測。記録するのは制限を掛ける前の変化量。"""
    mod = load_module(monkeypatch, PARC_ENSEMBLE=0, PARC_N_EXEC=1, PARC_ACT_SLEW=0.1)
    p = make_policy(mod, step_chunks([0.0, 0.5, 0.5]))
    p._warming = False

    drive(p, 3)

    assert len(p._d_xyz) == 2                # 先頭 step は基準が無いので数えない
    assert max(p._d_xyz) == pytest.approx(0.5)   # 0.1 に切った後の値ではない
    assert p._d_clip == 2


def test_delta_tracking_separates_translation_from_rotation(monkeypatch):
    """並進と回転は別の物理量。混ぜて 1 つの上限を引くのが ACT_SCALE の失敗。"""
    mod = load_module(monkeypatch, PARC_ENSEMBLE=0, PARC_N_EXEC=1)

    def chunks(i):
        a = np.zeros((50, 7), np.float32)
        if i:
            a[:, :3] = 0.4          # 並進だけ跳ねる
            a[:, 3:6] = 0.01
        return a

    p = make_policy(mod, chunks)
    p._warming = False
    drive(p, 2)

    assert p._d_xyz == pytest.approx([0.4])
    assert p._d_rot == pytest.approx([0.01])


def test_delta_tracking_runs_with_the_limiter_off(monkeypatch):
    """制限を入れる前に分布だけ見たいので、無効でも計測は回る。"""
    mod = load_module(monkeypatch, PARC_ENSEMBLE=0, PARC_N_EXEC=1)
    p = make_policy(mod, step_chunks([0.0, 0.3, 0.3]))
    p._warming = False

    drive(p, 3)

    assert len(p._d_xyz) == 2 and p._d_clip == 0
    assert max(p._d_xyz) == pytest.approx(0.3)


def test_delta_clip_counts_a_rotation_only_breach(monkeypatch):
    """回転だけ上限に当たった step も数える。"""
    mod = load_module(monkeypatch, PARC_ENSEMBLE=0, PARC_N_EXEC=1,
                      PARC_ACT_SLEW=0.5, PARC_ACT_SLEW_ROT=0.01)

    def chunks(i):
        a = np.zeros((50, 7), np.float32)
        if i:
            a[:, 3:6] = 0.2         # 回転だけ跳ねる。並進は上限 0.5 に届かない
        return a

    p = make_policy(mod, chunks)
    p._warming = False
    drive(p, 2)

    assert p._d_clip == 1


def test_delta_tracking_resets_between_episodes(monkeypatch):
    mod = load_module(monkeypatch, PARC_ENSEMBLE=0, PARC_N_EXEC=1)
    p = make_policy(mod, step_chunks([0.0, 0.3]))
    p._warming = False

    drive(p, 3)
    p.reset("next")

    assert p._d_xyz == [] and p._d_rot == [] and p._d_clip == 0


def test_delta_summary_reports_percentiles(monkeypatch):
    """mean と max だけでは外れ値が何 step あるか分からない。"""
    mod = load_module(monkeypatch)
    line = mod.MyPolicy._delta_desc("xyz", [0.01] * 90 + [0.2] * 10)

    assert "p50=0.0100" in line
    assert "max=0.2000" in line
    assert "p90=" in line and "p99=" in line


# --- 把持失敗後の開き直し ---------------------------------------------------
#
# BEHAVIOR-1K Challenge 2025 の 1 位が汎用の補正ルールとして入れたもの。
# 「掴み損なって閉じたまま、やり直さない」を検出して開かせる。


def grip(qpos):
    o = dict(OBS)
    o["robot0_gripper_qpos"] = np.array(qpos, np.float64)
    return o


def drive_obs(policy, obs, steps):
    return [float(policy.get_action(obs)[6]) for _ in range(steps)]


def closing_chunks(i):
    """常に閉じろと指令し続けるチャンク（環境規約で +1 = 閉じる）。"""
    a = np.zeros((50, 7), np.float32)
    a[:, 6] = 1.0
    return a


def test_grip_retry_is_off_by_default(monkeypatch):
    """未測定のつまみを既定で入れない。0.304 の構成を黙って変えないこと。"""
    cfg = load_module(monkeypatch).MyPolicy
    assert cfg.GRIP_RETRY == 0.0


def test_opening_is_the_sum_of_both_fingers(monkeypatch):
    """robosuite の Panda は左右で符号が逆。絶対値の和で開きを測る。"""
    mod = load_module(monkeypatch)
    assert mod.MyPolicy._grip_opening(grip([0.04, -0.04])) == pytest.approx(0.08)
    assert mod.MyPolicy._grip_opening(grip([0.001, -0.001])) == pytest.approx(0.002)
    assert mod.MyPolicy._grip_opening({}) is None


def test_empty_grasp_forces_the_gripper_open(monkeypatch):
    """閉指令が続いて指が閉じ切っていたら開き直す。"""
    mod = load_module(monkeypatch, PARC_ENSEMBLE=0, PARC_N_EXEC=1,
                      PARC_GRIP_RETRY=0.02, PARC_GRIP_RETRY_AFTER=3,
                      PARC_GRIP_RETRY_OPEN=4)
    p = make_policy(mod, closing_chunks)

    out = drive_obs(p, grip([0.001, -0.001]), 7)

    assert out[:2] == [1.0, 1.0]              # 過渡では手を出さない
    assert out[2:6] == [-1.0] * 4             # 3 step 目で判定して 4 step 開く
    assert out[6] == 1.0                      # その後は方策に戻す
    assert p._grip_retries == 1


def test_a_held_object_is_left_alone(monkeypatch):
    """物体を掴んでいれば指は物体の幅で止まる。そこには介入しない。"""
    mod = load_module(monkeypatch, PARC_ENSEMBLE=0, PARC_N_EXEC=1,
                      PARC_GRIP_RETRY=0.02, PARC_GRIP_RETRY_AFTER=3)
    p = make_policy(mod, closing_chunks)

    assert drive_obs(p, grip([0.015, -0.015]), 10) == [1.0] * 10
    assert p._grip_retries == 0


def test_retries_are_capped_per_episode(monkeypatch):
    """振動して 300 step 溶かすのを防ぐ。"""
    mod = load_module(monkeypatch, PARC_ENSEMBLE=0, PARC_N_EXEC=1,
                      PARC_GRIP_RETRY=0.02, PARC_GRIP_RETRY_AFTER=2,
                      PARC_GRIP_RETRY_OPEN=2, PARC_GRIP_RETRY_MAX=2)
    p = make_policy(mod, closing_chunks)

    drive_obs(p, grip([0.001, -0.001]), 60)

    assert p._grip_retries == 2


def test_measurement_runs_even_when_disabled(monkeypatch):
    """閾値を決めるための計測は、制限が無効でも回る（slew と同じ作法）。"""
    mod = load_module(monkeypatch, PARC_ENSEMBLE=0, PARC_N_EXEC=1)
    p = make_policy(mod, closing_chunks)
    p._warming = False

    drive_obs(p, grip([0.001, -0.001]), 5)

    assert len(p._grip_seen) == 5
    assert p._grip_seen[0] == pytest.approx(0.002)
    assert p._grip_retries == 0                # 無効なので介入はしない


def test_open_commands_are_not_counted(monkeypatch):
    """開指令中の開きは判定材料にしない（閉じていないのは当たり前）。"""
    mod = load_module(monkeypatch, PARC_ENSEMBLE=0, PARC_N_EXEC=1)

    def opening_chunks(i):
        a = np.zeros((50, 7), np.float32)
        a[:, 6] = -1.0
        return a

    p = make_policy(mod, opening_chunks)
    p._warming = False
    drive_obs(p, grip([0.001, -0.001]), 5)

    assert p._grip_seen == []


def test_grip_state_resets_between_episodes(monkeypatch):
    mod = load_module(monkeypatch, PARC_ENSEMBLE=0, PARC_N_EXEC=1,
                      PARC_GRIP_RETRY=0.02, PARC_GRIP_RETRY_AFTER=2)
    p = make_policy(mod, closing_chunks)
    p._warming = False
    drive_obs(p, grip([0.001, -0.001]), 6)

    p.reset("next")

    assert p._grip_seen == [] and p._grip_retries == 0
    assert p._grip_close_run == 0 and p._grip_open_left == 0


def test_retry_can_be_delayed_to_late_in_the_episode(monkeypatch):
    """物体の幅はタスクで違う。閾値を踏み外しても、後半に限れば壊す相手が減る。

    公開 4 タスクの実測では成功は 82〜187 step で終わり、空振りのまま
    握り続けた本は 300 step 使い切った。
    """
    mod = load_module(monkeypatch, PARC_ENSEMBLE=0, PARC_N_EXEC=1,
                      PARC_GRIP_RETRY=0.0025, PARC_GRIP_RETRY_AFTER=2,
                      PARC_GRIP_RETRY_MIN_STEP=10)
    p = make_policy(mod, closing_chunks)

    out = drive_obs(p, grip([0.0005, -0.0005]), 12)

    assert out[:10] == [1.0] * 10              # 10 step 目までは手を出さない
    assert out[10] == -1.0                     # そこを越えてから発火
    assert p._grip_retries == 1


def test_the_measured_threshold_separates_the_observed_episodes(monkeypatch):
    """公開 4 タスク 11 本の実測値。0.0025 がこの標本を完全に分離する。

    成功: 0.0037 0.0050 0.0536 0.0608 0.0630 0.0640 0.0641
    失敗: 0.0018（空振り）/ 0.0425（掴めているが別要因で失敗）

    余裕は 0.0018 と 0.0037 の間の 2 mm しかない。ここを動かすときは
    必ず測り直すこと。
    """
    mod = load_module(monkeypatch, PARC_ENSEMBLE=0, PARC_N_EXEC=1,
                      PARC_GRIP_RETRY=0.0025, PARC_GRIP_RETRY_AFTER=2)

    def fires(opening):
        p = make_policy(mod, closing_chunks)
        drive_obs(p, grip([opening / 2, -opening / 2]), 5)
        return p._grip_retries > 0

    assert fires(0.0018)                                   # 空振り -> 発火
    for ok in (0.0037, 0.0050, 0.0425, 0.0536, 0.0641):    # 掴めている -> 発火しない
        assert not fires(ok), ok
