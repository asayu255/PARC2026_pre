"""openvla-oft の `prismatic.vla.constants` の最小移植。

同梱の `modeling_prismatic.py`（checkpoint に付いてくる trust_remote_code の
コード）が import するのはこのモジュールと `prismatic.training.train_utils`
だけである。openvla-oft 本体を丸ごと入れる必要は無いので、必要な定数と
関数だけを写す。

**本家との唯一の違い**: 本家の constants.py は `detect_robot_platform()` で
`sys.argv` を見て LIBERO / ALOHA / BRIDGE を選び分ける。ここでは LIBERO 固定に
する。PARC の評価は LIBERO 系しか無く、コマンドラインを見て挙動が変わるのは
サーバープロセスでは事故のもとでしかない。

値は https://github.com/moojink/openvla-oft の LIBERO 設定に一致させてある。
"""
from enum import Enum


class NormalizationType(str, Enum):
    NORMAL = "normal"           # 平均 0 / 標準偏差 1
    BOUNDS = "bounds"           # [min, max] -> [-1, 1]
    BOUNDS_Q99 = "bounds_q99"   # [q01, q99] -> [-1, 1]


# --- 学習ラベル関連（推論では使わないが、import 時に解決される必要がある）---
IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 31743
STOP_INDEX = 2   # </s>

# --- LIBERO の設定 -----------------------------------------------------------
NUM_ACTIONS_CHUNK = 8
ACTION_DIM = 7
PROPRIO_DIM = 8
ACTION_PROPRIO_NORMALIZATION_TYPE = NormalizationType.BOUNDS_Q99
