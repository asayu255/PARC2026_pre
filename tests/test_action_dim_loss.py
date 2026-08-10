"""tools/action_dim_loss.py の単体テスト。

守りたいのは 2 つ。

- ゼロ埋め次元へ**勾配が流れない**こと。ここが 25/32 = 78% を占めていて、
  目標が入力から決定的に復元できる自明な項である（§34.7）
- 残した次元の**損失のスケールが変わらない**こと。SmolVLAPolicy.forward は
  返り値を 32 次元で平均するので、7/32 に縮むと実効学習率が 4.6 分の 1 に
  なってしまう。それでは「損失を絞ったから」なのか「学習率が下がったから」
  なのか分からなくなる

lerobot は要らない。VLAFlowMatching を偽物に差し替えて確かめる。
"""
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "tools"))

import action_dim_loss  # noqa: E402

TOTAL_DIM = 32
REAL_DIM = 7


def fake_lerobot(monkeypatch, losses):
    """VLAFlowMatching.forward が `losses` を返す偽モジュールを立てる。"""

    class FakeFlowMatching:
        def forward(self, *args, **kwargs):
            return losses

    mod = types.ModuleType("lerobot.policies.smolvla.modeling_smolvla")
    mod.VLAFlowMatching = FakeFlowMatching
    for name, parent in (
        ("lerobot", types.ModuleType("lerobot")),
        ("lerobot.policies", types.ModuleType("lerobot.policies")),
        ("lerobot.policies.smolvla", types.ModuleType("lerobot.policies.smolvla")),
        ("lerobot.policies.smolvla.modeling_smolvla", mod),
    ):
        monkeypatch.setitem(sys.modules, name, parent)
    return FakeFlowMatching


def test_padded_dims_are_removed_from_the_loss(monkeypatch):
    losses = torch.ones(2, 50, TOTAL_DIM)
    cls = fake_lerobot(monkeypatch, losses)
    action_dim_loss.install_patch(REAL_DIM, log_every=0)

    out = cls().forward()

    assert torch.all(out[..., REAL_DIM:] == 0)
    assert torch.all(out[..., :REAL_DIM] > 0)


def test_scale_matches_a_plain_mean_over_real_dims(monkeypatch):
    """SmolVLAPolicy.forward の `.mean()` を通したあとの値が、
    本物の次元だけの平均と一致すること（実効学習率を変えない）。"""
    torch.manual_seed(0)
    losses = torch.rand(4, 50, TOTAL_DIM)
    cls = fake_lerobot(monkeypatch, losses)
    action_dim_loss.install_patch(REAL_DIM, log_every=0)

    patched_mean = cls().forward().mean()          # lerobot 側の 32 次元平均
    real_only_mean = losses[..., :REAL_DIM].mean()

    assert torch.allclose(patched_mean, real_only_mean, atol=1e-6)


def test_no_gradient_reaches_the_padded_dims(monkeypatch):
    """本題。ゼロ埋め次元は勾配を受け取らない。"""
    source = torch.rand(2, 50, TOTAL_DIM, requires_grad=True)
    cls = fake_lerobot(monkeypatch, source * 1.0)
    action_dim_loss.install_patch(REAL_DIM, log_every=0)

    cls().forward().mean().backward()

    assert torch.all(source.grad[..., REAL_DIM:] == 0)
    assert torch.all(source.grad[..., :REAL_DIM] != 0)


def test_noop_when_there_is_no_padding(monkeypatch):
    losses = torch.rand(2, 50, REAL_DIM)
    cls = fake_lerobot(monkeypatch, losses)
    action_dim_loss.install_patch(REAL_DIM, log_every=0)

    assert cls().forward() is losses


def test_stats_split_real_and_padded(monkeypatch):
    losses = torch.cat(
        [torch.full((1, 4, REAL_DIM), 2.0), torch.full((1, 4, TOTAL_DIM - REAL_DIM), 0.5)],
        dim=-1,
    )
    cls = fake_lerobot(monkeypatch, losses)
    stats = action_dim_loss.install_patch(REAL_DIM, log_every=0)

    cls().forward()

    assert stats.n == 1
    assert stats.real == pytest.approx(2.0)
    assert stats.pad == pytest.approx(0.5)
    assert "2.0000" in stats.summary() and "0.5000" in stats.summary()


def test_summary_reports_the_padding_share(monkeypatch):
    """素の 32 次元平均のうち、ゼロ埋めが何 % を占めるかを出す。"""
    stats = action_dim_loss.DimLossStats()
    stats.add(real=1.0, pad=1.0)          # 全次元が同じ損失なら
    # ゼロ埋めは 25/32 = 78%
    assert "78%" in stats.summary()
