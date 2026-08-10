"""学習損失を「本物の action 次元」だけに絞る。

なぜ要るか
----------
SmolVLA は action を `max_action_dim`（既定 **32**）までゼロ埋めする。
PARC の action は **7 次元**なので、25 次元がゼロ埋めである。

    lerobot/policies/smolvla/modeling_smolvla.py
      prepare_action : actions = pad_vector(batch[ACTION], 32)
      VLAFlowMatching.forward
        x_t    = t * noise + (1 - t) * actions
        u_t    = noise - actions
        losses = F.mse_loss(u_t, v_t, reduction="none")   # (B, chunk, 32)
      SmolVLAPolicy.forward
        losses = losses[:, :, : self.config.max_action_dim]   # ← 32 で切るので no-op
        loss   = losses.mean()                                # ← 32 次元の平均

`# Remove padding` というコメントが付いているが、`max_action_dim` は
ゼロ埋め**後**の次元なので、この行は何も削っていない。損失は
**32 次元すべての平均**である。

ゼロ埋め次元では `actions_d = 0` なので

    x_t,d = t * noise_d
    u_t,d = noise_d - 0 = noise_d = x_t,d / t

つまり目標値が**入力から決定的に復元できる**。時刻 t もネットワークに
入っているので、これは自明に学習できる関数であり、タスクとは無関係である。

結果として **損失の 25/32 = 78% が、タスクと無関係で簡単な項**になる。
rank 8 の LoRA は容量が小さいので、勾配の大半が向くこの項に容量を使い、
本物の 7 次元は巻き添えで悪化しうる。これは実測と整合する（§34.5）。

    学習 loss 0.93 -> 0.30（3 分の 1）  /  success 77.5% -> 32.5%
    画素数を 1/4 にしても loss は 2% しか動かない（画像に依存しない項が主）
    画像スロットを 5->2 にしても loss が 12 桁一致（§27.1 の仮説 1）

このモジュールは損失をゼロ埋め次元へ流さないようにする。ベースモデルの
学習がどうだったかは分からないが、**少なくとも追加学習の勾配は
タスクに向く**ようになる。
"""

from __future__ import annotations

import torch

#: PARC / LIBERO の action 次元 [dx, dy, dz, droll, dpitch, dyaw, gripper]
PARC_ACTION_DIM = 7


class DimLossStats:
    """本物の次元とゼロ埋め次元それぞれの損失を覚えておく（診断用）。"""

    def __init__(self) -> None:
        self.n = 0
        self.real = 0.0
        self.pad = 0.0

    def add(self, real: float, pad: float) -> None:
        self.n += 1
        self.real += real
        self.pad += pad

    def summary(self) -> str:
        if not self.n:
            return "（まだデータなし）"
        real, pad = self.real / self.n, self.pad / self.n
        total = (real * PARC_ACTION_DIM + pad * (32 - PARC_ACTION_DIM)) / 32
        share = (pad * (32 - PARC_ACTION_DIM) / 32) / total * 100 if total else float("nan")
        return (
            f"本物の {PARC_ACTION_DIM} 次元: {real:.4f}   "
            f"ゼロ埋め次元: {pad:.4f}   "
            f"（素の 32 次元平均なら {total:.4f}、うちゼロ埋めが {share:.0f}%）"
        )

    def reset(self) -> None:
        self.__init__()


def install_patch(action_dim: int = PARC_ACTION_DIM, log_every: int = 100):
    """VLAFlowMatching.forward の戻り値からゼロ埋め次元を落とす。

    `SmolVLAPolicy.forward` は返ってきた損失を 32 次元で平均するので、
    残す次元を `32 / action_dim` 倍しておく。こうすると
    「本物の次元だけの平均」と同じ値・同じ勾配スケールになり、
    学習率をいじらずに済む。

    Returns:
        DimLossStats: 直近の内訳。学習ループ側から読める。
    """
    from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching

    stats = DimLossStats()
    original_forward = VLAFlowMatching.forward
    calls = {"n": 0}

    def patched_forward(self, *args, **kwargs):
        losses = original_forward(self, *args, **kwargs)
        total_dim = losses.shape[-1]
        if action_dim >= total_dim:
            return losses                      # 埋めが無い。触らない

        with torch.no_grad():
            stats.add(
                float(losses[..., :action_dim].mean()),
                float(losses[..., action_dim:].mean()),
            )
        calls["n"] += 1
        if log_every and calls["n"] % log_every == 1:
            print(f"[dimloss] {stats.summary()}", flush=True)
            stats.reset()

        mask = losses.new_zeros(total_dim)
        mask[:action_dim] = total_dim / action_dim
        return losses * mask

    VLAFlowMatching.forward = patched_forward
    print(
        f"[dimloss] 損失を先頭 {action_dim} 次元に限定する"
        f"（ゼロ埋めの {32 - action_dim} 次元を勾配から外す）",
        flush=True,
    )
    return stats
