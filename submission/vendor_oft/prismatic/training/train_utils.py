"""openvla-oft の `prismatic.training.train_utils` の最小移植。

`modeling_prismatic.py` が import するのはこの 2 関数だけである。
学習ラベルから action トークンの位置を切り出すためのもので、推論経路では
呼ばれないが、モジュールの import 時に解決される必要がある。

https://github.com/moojink/openvla-oft の実装をそのまま写した。
"""
import torch

from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX


def get_current_action_mask(token_ids):
    newline_positions = token_ids != IGNORE_INDEX
    cumsum = torch.cumsum(newline_positions, dim=1)
    mask = (1 <= cumsum) & (cumsum <= ACTION_DIM)
    action_tokens_only_mask = token_ids > ACTION_TOKEN_BEGIN_IDX
    mask = action_tokens_only_mask * mask
    return mask


def get_next_actions_mask(token_ids):
    newline_positions = token_ids != IGNORE_INDEX
    cumsum = torch.cumsum(newline_positions, dim=1)
    mask = cumsum > ACTION_DIM
    action_tokens_only_mask = token_ids > ACTION_TOKEN_BEGIN_IDX
    mask = action_tokens_only_mask * mask
    return mask
