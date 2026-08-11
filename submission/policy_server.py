"""ポリシーサーバー（提出物）

submission_template/policy_server.py から派生した作業用の提出物である。
テンプレートはハーネスのテストフィクスチャも兼ねているため未編集のまま残し、
実装はこちらに置く。編集してよいのは MyPolicy クラスの中身だけで、
それ以外のコード（サーバー部分、シリアライゼーション）は変更不可である。

ローカルテスト:
    pip install -r requirements.txt
    python policy_server.py                  # サーバー起動（port 8000）

    # 別ターミナルで評価実行（リポジトリ直下から）
    python -m pipeline --server-url http://localhost:8000 --track track1 \
        --n-episodes 2 --max-steps 600

提出前チェック（リポジトリ直下から）:
    python validate_submission.py submission/
"""

import argparse
import os
import time
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path

import msgpack
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response


# ============================================================
# ポリシーのインターフェース定義（変更不可）
# MyPolicy が満たすべき get_action() / reset() の仕様を定める。
# ============================================================


class BasePolicy(ABC):
    """ポリシーの基底クラス。get_action() と reset() を実装してください。"""

    @abstractmethod
    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """観測からアクションを推論する。

        Args:
            obs: 環境からの観測。以下のキーが含まれる:
                - "agentview_image": (128, 128, 3) uint8
                - "robot0_eye_in_hand_image": (128, 128, 3) uint8
                - "robot0_joint_pos": (7,) float
                - "robot0_eef_pos": (3,) float
                - "robot0_eef_quat": (4,) float
                - "robot0_gripper_qpos": (2,) float

        Returns:
            action: (7,) float32 — [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        ...

    @abstractmethod
    def reset(self, instruction: str = "") -> None:
        """エピソード開始時に呼ばれる。内部状態をリセットしてください。

        Args:
            instruction: タスクの言語指示（例: "pick up the red mug and place it on the shelf"）
        """
        ...


# ============================================================
# ここを編集する（MyPolicy の中身だけを自分のモデルに置き換える）
# ============================================================


_HERE = Path(__file__).resolve().parent

# 既定は同梱の model_weights。追加学習したモデルを評価するときは
# PARC_WEIGHTS_DIR でマージ済みディレクトリを指せば、ファイルを
# 入れ替えずに A/B が取れる。提出 zip では未設定なので影響しない。
_WEIGHTS_DIR = Path(os.environ.get("PARC_WEIGHTS_DIR") or (_HERE / "model_weights"))

# 採点環境は外部通信が無いので、tokenizer / VLM バックボーンもローカルに置く。
# SmolVLA 以外のポリシー（π0 系は PaliGemma の tokenizer を要する）を試すときは
# PARC_BACKBONE_DIR で別ディレクトリを指す。提出 zip では未設定。
_BACKBONE_DIR = Path(os.environ.get("PARC_BACKBONE_DIR") or (_HERE / "smolvlm_backbone"))

# lerobot は pip では入れられないので同梱する。
# lerobot の必須依存に pynput があり、Linux ではこれが evdev を引く。
# evdev は wheel が一切公開されておらず必ずソースビルドになるが、
# 採点環境には Python.h (python3-dev) が無いためコンパイルに失敗する。
#
#   pynput -> evdev -> C 拡張 -> fatal error: Python.h: No such file or directory
#
# 推論の import 連鎖に pynput/evdev は含まれない（lerobot が実機操作用に
# 宣言しているだけ）。そこで lerobot 本体を vendor/ へ置き、requirements.txt
# には wheel のある依存だけを列挙する。
_VENDOR_DIR = _HERE / "vendor"

# --- 切り分け用の計装（PARC_DEBUG_DIR を設定したときだけ有効。既定は完全に無効）---
_DEBUG_DIR = os.environ.get("PARC_DEBUG_DIR")


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    """環境変数を整数で読む。不正値は既定へ落として警告する。

    実験用のつまみを環境変数で振れるようにするためのもの。値が壊れていても
    サーバーが起動しないより既定で動くほうがよい（提出時は未設定なので既定）。
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        v = int(raw)
    except ValueError:
        print(f"[MyPolicy] {name}={raw!r} を整数として読めない。既定 {default} を使う。")
        return default
    if v < minimum:
        print(f"[MyPolicy] {name}={v} は下限 {minimum} 未満。{minimum} に切り上げる。")
        return minimum
    return v


def _env_float(name: str, default: float) -> float:
    """環境変数を float で読む。不正値は既定へ落として警告する。

    _env_int と違い下限は設けない。temporal ensembling の減衰係数は
    負の値にも意味がある（新しい予測を重く見る向きになる）。
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[MyPolicy] {name}={raw!r} を float として読めない。既定 {default} を使う。")
        return default


_DEBUG_MAX = _env_int("PARC_DEBUG_MAX", 3)


class MyPolicy(BasePolicy):
    """SmolVLA (lerobot/smolvla_libero_plus) ポリシー。

    model_weights/ が無い場合はゼロ action の静止ベースラインとして動作し、
    評価パイプラインの疎通確認に使える。

    入出力の仕様は checkpoint の実物から確定させたものである（推測ではない）。

    policy_preprocessor.json の rename_observations_processor:
        observation.images.front -> observation.images.camera1
        observation.images.wrist -> observation.images.camera2
    したがって渡すキーは front / wrist であり camera1 / camera2 ではない。

    normalizer の observation.state 統計は 8 次元で、値域から
        [0:3] eef_pos       (±0.5 程度, m)
        [3:6] axis_angle    (±3.8 程度, rad)
        [6:8] gripper_qpos  (±0.042, std 0.014)
    と読める。config.json の input_features は [6] と宣言しているが、
    NormalizerProcessorStep._apply_transform は統計をスライスせず
    (tensor - mean) / std をそのまま適用するため、統計側の次元に合わせる。
    次元は起動時に統計から自動判定する（_detect_state_dim）。

    camera3 / empty_camera_0 / empty_camera_1 は渡さない。
    SmolVLAPolicy.prepare_images が不足キーを -1 埋めの空画像で補い
    (config.empty_cameras=2)、normalizer は存在するキーのみ処理する。
    学習時も front / wrist のみだったため条件は一致する。

    画像は float32 CHW の [0, 1] で渡す。512 へのリサイズと [-1, 1] への
    変換は prepare_images がモデル内部で行うので、ここでやってはいけない。
    """

    # --- 提出物の観測 -> SmolVLA の入力キー -----------------------------------
    # 実際に使うキーは起動時に policy_preprocessor.json の rename_map から
    # 導出する（_resolve_image_keys）。追加学習でキー名が変わっても追従する。
    # 以下は導出できなかった場合のフォールバック既定。
    KEY_MAIN = "observation.images.front"    # agentview   -> camera1
    KEY_WRIST = "observation.images.wrist"   # eye_in_hand -> camera2
    KEY_STATE = "observation.state"

    #: 手先カメラと判定する語（キー名に含まれていれば wrist 扱い）
    WRIST_HINTS = ("wrist", "eye_in_hand", "in_hand", "hand", "gripper")

    #: LIBERO の生画像を LeRobot 側の向きへ揃える 180 度回転。
    #: LiberoProcessorStep (lerobot/processor/env_processor.py) の
    #: torch.flip(img, dims=[2, 3]) と同じ変換である。
    #:
    #: 実測で確定済み。put_the_bowl_on_the_stove_light_11 を 150 ステップ
    #: 走らせ、対象物体への最接近距離を比較した結果:
    #:     True : bowl 0.078 (最接近 step 41) < wine 0.102 < cream 0.138
    #:     False: wine 0.166 < bowl 0.180 < cream 0.185
    #: True では指示された bowl が最も近く軌跡も収束するが、False では
    #: 3 物体が 0.166-0.185 に団子になり選択性が失われる。対象への接近は 2.3 倍差。
    #:
    #: A/B 用に PARC_FLIP180=0 で無効化できる。既定（未設定）は有効。
    FLIP_IMAGES_180 = os.environ.get("PARC_FLIP180", "1") != "0"

    #: config.json の chunk_size / n_action_steps に一致させる
    ACTION_CHUNK_SIZE = 50

    #: 生成したチャンクのうち実際に環境へ流すステップ数。
    #: 小さくすると再推論の頻度が上がり閉ループ性が増す。open-loop 区間が
    #: 長いほど、対象外の物体へ接触する余地が増える（衝突判定は 1mm）。
    #:
    #: 学習時の設定は 50（config の n_action_steps）だが、実測では下げた方が
    #: 良い。ただし下げ続ければ良いわけではなく、5 付近に最小がある。
    #: put_the_bowl_on_the_stove_light_11 / 300 step / EGL / 各 50 エピソード:
    #:     n_exec  success  collision  cartesian  jerk(rms)  sparc
    #:        10     0.84      0.120      0.842      6.470   -2.421
    #:         5     0.92      0.080      0.838      6.091   -2.371
    #:         2     0.80      0.180      0.826      6.899   -2.407
    #:
    #: success の 0.92 vs 0.84 は 1.24σ で単独では有意でない。決め手は jerk が
    #: 独立な 2 回（OSMesa 10 エピソード / EGL 50 エピソード）で同じ順序を
    #: 再現したことで、値も 6.477->6.470、6.140->6.091、7.121->6.899 とほぼ
    #: 一致した。jerk は 1 エピソード約 90 step ×本数の平均で分散が小さい。
    #: その上で success と collision も同じ向きを指した。
    #:
    #: 中間に最小ができる理由: 短くするほどチャンク境界での再計画が増え、
    #: flow-matching は毎回ノイズを引き直すので境界で不連続が出る。逆に
    #: 長すぎると誤差が溜まって補正が跳ねる。
    #:
    #: 推論は 1 回 0.24 秒、採点実測でも max 0.455 秒（制限 10 秒）なので、
    #: 5 に下げても余裕は十分ある。
    #:
    #: A/B 用に PARC_N_EXEC で上書きできる。既定（未設定）は 5。
    #:
    #: なお TEMPORAL_ENSEMBLE が既定で有効になったため、この値は
    #: `PARC_ENSEMBLE=0` で ensembling を切ったときにしか使われない。
    N_ACTION_EXEC = _env_int("PARC_N_EXEC", 5)

    # --- temporal ensembling (ACT / Zhao et al. 2023) -------------------------
    #: n_exec を下げても改善しなかった理由は「open-loop 区間が長いこと」では
    #: なく「チャンク境界の不連続」である（n_exec=2 は 3 指標すべてで最悪）。
    #: flow-matching は推論のたびにノイズを引き直すので、境界で前後のチャンクが
    #: 食い違う。区間を短くすると境界の数が増えるだけで、悪化する。
    #:
    #: temporal ensembling は境界で切り替える代わりに混ぜる。毎ステップ推論し、
    #: 「そのステップ向けに過去 H 回の推論が出した予測」を指数重みで平均する。
    #: 再サンプリング由来のばらつきが平均で落ちるので、モデルには一切触れずに
    #: jerk（＝まさにこの量を測っている指標）を下げられる。
    #:
    #: 有効にすると N_ACTION_EXEC は使われない（推論頻度は ENSEMBLE_QUERY_EVERY）。
    #:
    #: **既定で有効。** 独立な 2 ラウンドで jerk が再現した。
    #:
    #:     ラウンド              base    h8(採用)   h16new    h16(ACT そのまま)
    #:     stove 50ep           5.932   3.632      3.651     4.129
    #:     公開 4 タスク 40ep    7.256   4.508      4.232     —
    #:
    #: タスクも本数も違って −38〜42% が揃っている。同一条件の再測定での
    #: jerk のぶれは 2.6% なので、これは実効果である。
    #:
    #: ただし **collision は動かなかった**（4 タスクで 3 条件とも 0.075 の同値）。
    #: 狙いは衝突の低減だったので、その意味では外れている。それでも採用するのは、
    #: 採点が 8 本中 成功 0・到達 2 で、0.077 がほぼ全部 軌跡系の部分点だから
    #: である（§22.3）。いま点になっている量そのものを 38% 改善する。
    #: 失敗モード（未到達 6 本・衝突 2 本）はどちらも直らない。
    #:
    #: 劣化は無い。success は 2 ラウンドで +4 pt / −5 pt と方向が定まらず
    #: いずれも有意でない。平均ステップは 4 タスクすべてで微減しており、
    #: 「差分を平均すると動きが縮んで遅くなる」懸念は起きていない。
    #:
    #: A/B 用に PARC_ENSEMBLE=0 で従来の N_ACTION_EXEC 経路へ戻せる。
    TEMPORAL_ENSEMBLE = os.environ.get("PARC_ENSEMBLE", "1") != "0"

    #: 各チャンクから何ステップ先までを ensembling に積むか。
    #: 同時に重なる予測の本数は概ね H / ENSEMBLE_QUERY_EVERY になる。
    #:
    #: ACT はチャンク全長（100）を積むが、あちらの action は関節の絶対位置で、
    #: 古い予測でも「同じ目標姿勢」の推定値なので陳腐化しにくい。PARC の action
    #: は差分（dx, dy, dz, ...）なので、H 歩前の観測に基づく予測は「今どこに
    #: いるか」の想定がずれたぶん系統的に外れる。
    #:
    #: 実測でもそのとおりで、16 本を ACT の重みでほぼ一様に混ぜる h16 は
    #: jerk 4.129 にとどまり、最近の予測を重く見る 2 通り（h8 = 古いものを
    #: 捨てる / h16new = 古いものの重みを下げる）が 3.63 / 3.65 で並んだ。
    #: **ACT の重み付けをそのまま持ってくるのは、この問題設定では最適ではない。**
    #:
    #: h8 と h16new は決着しなかった（順序がラウンド間で入れ替わる）。h8 を
    #: 採ったのは測定ではなく構造の理由で、バッファが半分なので古い差分予測を
    #: 抱える量が構造的に小さく、エピソード冒頭の立ち上がりも 8 step で済む。
    ENSEMBLE_HORIZON = _env_int("PARC_ENS_H", 8)

    #: 何ステップおきに推論するか。1 が ACT 本来の設定（毎ステップ）。
    #: 2 にすると推論回数は半分、重なる予測の本数も半分になる。
    #:
    #: 1 リクエストの制約には影響しない。推論 1 回のコストは変わらないので、
    #: 実測でも /act の最悪値は 0.30 秒（制限 10 秒）で、n_exec=5 のときの
    #: 0.366 秒より小さかった（GPU が連続稼働で温まるため）。
    #: 効くのは合計時間だけで、3 回目の採点 269 秒に対し 950 秒前後になる。
    #: README のとおり累積の制限は無いが、詰めたい場合はここを 2 にする
    #: （jerk への影響は未測定）。
    ENSEMBLE_QUERY_EVERY = _env_int("PARC_ENS_QUERY", 1)

    #: 重み w_i = exp(-m * i)。i は予測の古い順の番号で、i=0 が最も古い。
    #: ACT の実装（k=0.01）と同じ向き・同じ既定値である。
    #:
    #: この向きは ACT が絶対位置を出力することを前提にしている。差分 action では
    #: 古い予測ほど陳腐化するので、**m を負にして新しい予測を重く見る**ほうが
    #: 筋が通る可能性がある。符号を含めて A/B で決めること。
    #:   m = +0.01 … ほぼ一様平均（ACT 既定。50 本でも最古:最新 = 1:0.61）
    #:   m = 0     … 完全な一様平均
    #:   m = -0.1  … 新しい予測を重く見る（16 本で最新:最古 = 1:0.22）
    ENSEMBLE_M = _env_float("PARC_ENS_M", 0.01)

    #: gripper（action[6]）も平均するか。0 にすると最新の予測をそのまま使う。
    #:
    #: ACT は全次元を平均するが、あちらの gripper は連続的な開度である。
    #: LIBERO の gripper は実質 2 値（±1）なので、平均すると開閉の切り替わりが
    #: 数ステップ鈍る。把持の遅れが効くなら 0 側が有利になりうる。
    ENSEMBLE_GRIPPER = os.environ.get("PARC_ENS_GRIPPER", "1") != "0"

    #: この秒数を超えた /act を警告する。評価側の打ち切りは 10 秒で、
    #: 1 回でも超えるとトラック全体が 0 点になる（README のタイムアウト仕様）。
    #: 推論は実測 0.24 秒だが、n_exec=50 の評価で 10 秒超のストールが実際に
    #: 起きてトラックが落ちた。外れ値を取りこぼさないよう常時計測する。
    SLOW_REQUEST_SEC = float(os.environ.get("PARC_SLOW_SEC", "2.0"))

    def __init__(self):
        self.instruction = ""
        self._queue: deque[np.ndarray] = deque()
        # temporal ensembling の生きている予測。要素は [消費済みステップ数, (H,7)]。
        # 全要素の H が同じなので、古いものから順に尽きる = FIFO で捨てられる。
        self._ens: deque[list] = deque()
        self._step = 0
        self.torch = None
        self.device = None
        self.preprocessor = None
        self.postprocessor = None
        self.state_dim = 8
        self.key_main = self.KEY_MAIN
        self.key_wrist = self.KEY_WRIST
        self._dbg_n = 0
        self._trace_i = 0
        self._warming = False
        self._lat_max = 0.0
        self._lat_sum = 0.0
        self._lat_n = 0
        self._lat_slow = 0
        self.policy_type = "smolvla"   # _load_model が config から上書きする
        self.oft = None                # OpenVLA-OFT のときだけ入る
        self.model = self._load_model()
        self._warmup()

    # ------------------------------------------------------------
    # モデルのロードと推論
    # ------------------------------------------------------------

    def _load_model(self):
        """SmolVLA と保存済み processor をロードする。

        サーバー起動から /health が 200 を返すまでの制限は 120 秒である。
        実測のロード時間は約 7 秒。
        """
        if not _WEIGHTS_DIR.is_dir():
            print(
                f"[MyPolicy] {_WEIGHTS_DIR} が無いため、ゼロ action の"
                " ベースラインで動作する（成功率は 0 になる）。"
                f"\n[MyPolicy]   ensemble={self._ensemble_desc()}"
            )
            return None

        # OpenVLA-OFT は lerobot をまったく使わない（使えない。checkpoint は
        # transformers 4.40.1 前提で、lerobot 0.4.4 は SmolVLA 用に 4.57.1 以上を
        # 要求する）。lerobot を sys.path へ入れる前に分岐する。
        import oft_policy

        if oft_policy.is_oft_checkpoint(_WEIGHTS_DIR):
            return self._load_oft_model(oft_policy)

        # 同梱した lerobot を優先する。site-packages に別バージョンが
        # 入っていても、こちらが先に解決される。
        if _VENDOR_DIR.is_dir():
            import sys
            v = str(_VENDOR_DIR)
            if v not in sys.path:
                sys.path.insert(0, v)
            print(f"[MyPolicy] vendor: {_VENDOR_DIR}")

        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.torch = torch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        cfg = PreTrainedConfig.from_pretrained(str(_WEIGHTS_DIR), local_files_only=True)

        # ポリシー種別は config.json の type から引く。SmolVLA を決め打ちしない。
        # 以降の processor / 画像キー / state 次元の解決はすべて保存済み設定から
        # 導いているので、重みを差し替えるだけで別のポリシーを評価できる。
        self.policy_type = getattr(cfg, "type", "smolvla")
        policy_cls = get_policy_class(self.policy_type)

        # 採点環境は外部通信が無く HF キャッシュも存在しない。
        # バックボーンを同梱している場合はローカルパスを見るよう差し替える。
        # vlm_model_name を持つのは SmolVLA だけで、π0 系は VLM を config から
        # 構築する（重みは checkpoint に入っている）ため、この差し替えは不要。
        # 無い属性を勝手に生やさないよう、存在を見てから触る。
        if not hasattr(cfg, "vlm_model_name"):
            print(f"[MyPolicy] {self.policy_type}: vlm_model_name を持たないので差し替え不要")
        elif _BACKBONE_DIR.is_dir():
            cfg.vlm_model_name = str(_BACKBONE_DIR)
            print(f"[MyPolicy] VLM backbone: {_BACKBONE_DIR}")
        else:
            print(
                f"[MyPolicy] 警告: {_BACKBONE_DIR} が無い。"
                f" vlm_model_name={getattr(cfg, 'vlm_model_name', '?')} を"
                " HF キャッシュから解決するため、採点環境では起動に失敗する。"
            )

        # flow matching の積分ステップ数（config の既定は 10）。
        # sample_actions は速度場を Euler 法で 1 -> 0 まで積分する
        # （modeling_smolvla.py: dt = -1/num_steps のループ）。増やすほど
        # 積分誤差が小さくなり、同じモデルからより正確な action が出る。
        #
        # VLM の prefix は KV キャッシュされてループの外なので、増える計算は
        # action expert の denoise_step だけである。レイテンシは採点実測で
        # max 0.463 秒（制限 10 秒に対し 21 倍の余裕）なので予算はある。
        #
        # 既定は config のまま（＝挙動を変えない）。A/B は PARC_NUM_STEPS で行う。
        steps_override = _env_int("PARC_NUM_STEPS", 0, minimum=0)
        if steps_override:
            print(
                f"[MyPolicy] num_steps: {getattr(cfg, 'num_steps', '?')}"
                f" -> {steps_override}"
            )
            cfg.num_steps = steps_override

        if self.policy_type in ("pi0", "pi05", "pi0_fast"):
            self._ensure_siglip_check()

        model = policy_cls.from_pretrained(
            str(_WEIGHTS_DIR), config=cfg, local_files_only=True
        )
        model = model.eval().to(self.device)
        # from_pretrained が config を読み直す場合に備えて、実体側でも上書きを確認する。
        if steps_override and getattr(model.config, "num_steps", None) != steps_override:
            model.config.num_steps = steps_override
            print(f"[MyPolicy] num_steps をモデル側にも適用: {steps_override}")

        import lerobot
        print(f"[MyPolicy] lerobot: {Path(lerobot.__file__).parent}")

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=str(_WEIGHTS_DIR),
            preprocessor_overrides=self._preprocessor_overrides(),
        )

        self._resolve_image_keys()
        self.state_dim = self._detect_state_dim()
        if self.state_dim != 6 and self.state_dim < 8:
            raise RuntimeError(
                f"observation.state の次元 {self.state_dim} に対応する構成が不明。"
                " 6 (eef_pos+axis_angle) か 8 以上 (+gripper_qpos、余りはゼロ埋め)"
                " のみ対応する。"
            )
        if self.state_dim > 8:
            # π0 系は max_state_dim=32 を宣言しており、統計もその次元で
            # 保存されている場合がある。lerobot 内部の pad_vector と同じく
            # 後ろをゼロで埋める（_to_state 側で行う）。
            print(f"[MyPolicy] state を 8 -> {self.state_dim} へゼロ埋めして渡す")

        print(
            f"[MyPolicy] weights: {_WEIGHTS_DIR}"
            f"{' (PARC_WEIGHTS_DIR)' if os.environ.get('PARC_WEIGHTS_DIR') else ''}"
            f"\n[MyPolicy] {self.policy_type} ready | device={self.device}"
            f" | state_dim={self.state_dim} | chunk={self.ACTION_CHUNK_SIZE}"
            f" | exec={self.N_ACTION_EXEC} | flip180={self.FLIP_IMAGES_180}"
            f"\n[MyPolicy]   main={self.key_main} wrist={self.key_wrist}"
            f"\n[MyPolicy]   ensemble={self._ensemble_desc()}"
            f" | num_steps={getattr(model.config, 'num_steps', '?')}"
        )
        return model

    def _load_oft_model(self, oft_policy):
        """OpenVLA-OFT+ を読み込む。lerobot 側の経路とは完全に別。

        SmolVLA 側で config から導いていたもの（画像キー、state 次元、
        processor）は OFT では固定である。LIBERO の観測は主カメラ + 手首の
        2 枚で、state は eef_pos(3) + axis_angle(3) + gripper_qpos(2) の 8 次元。
        """
        import torch

        self.torch = torch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.policy_type = "openvla-oft"
        self.state_dim = 8
        self.ACTION_CHUNK_SIZE = oft_policy.NUM_ACTIONS_CHUNK

        # 提出 zip では環境変数を一切設定できないので、ここの既定がそのまま
        # 採点で走る構成になる。SmolVLA 用の既定（ensembling on / gripper 平均）
        # をそのまま使うと、一度も測っていない構成で採点されることになる。
        #
        # OFT の既定は本家 run_libero_eval の GenerateConfig に合わせる
        # （num_open_loop_steps=8、ensembling なし）。LIBERO-Plus で 79.6 が
        # 出たのはその構成であり、こちらが勝手に足したものではない。
        #
        # gripper を平均してはいけない。OFT の gripper は sign() で ±1 に
        # 二値化されているため、平均すると中間値になって二値化が壊れる。
        # ensembling を使う場合も最新の予測をそのまま採る。
        if "PARC_ENSEMBLE" not in os.environ:
            self.TEMPORAL_ENSEMBLE = False
            self.N_ACTION_EXEC = oft_policy.NUM_ACTIONS_CHUNK
            print(
                "[MyPolicy] OFT の既定として ensembling を無効にし"
                f" exec={self.N_ACTION_EXEC} にする（本家 num_open_loop_steps）"
            )
        if "PARC_ENS_GRIPPER" not in os.environ:
            self.ENSEMBLE_GRIPPER = False

        gripper = os.environ.get("PARC_OFT_GRIPPER", "binarize")
        self.oft = oft_policy.OFTModel(
            _WEIGHTS_DIR,
            device=str(self.device),
            flip180=self.FLIP_IMAGES_180,
            center_crop=oft_policy.env_flag("PARC_OFT_CENTER_CROP", True),
            unnorm_key=os.environ.get("PARC_OFT_UNNORM") or None,
            gripper_transform=gripper != "off",
            gripper_binarize=gripper != "linear",
        )
        print(
            f"[MyPolicy] weights: {_WEIGHTS_DIR}"
            f"{' (PARC_WEIGHTS_DIR)' if os.environ.get('PARC_WEIGHTS_DIR') else ''}"
            f"\n[MyPolicy] {self.policy_type} ready | device={self.device}"
            f" | state_dim={self.state_dim} | chunk={self.ACTION_CHUNK_SIZE}"
            f" | flip180={self.FLIP_IMAGES_180}"
            f"\n[MyPolicy]   ensemble={self._ensemble_desc()}"
        )
        return self.oft

    @staticmethod
    def _ensure_siglip_check() -> None:
        """π0 系が起動時に要求する transformers.models.siglip.check を用意する。

        lerobot 0.4.4 の π0 / π0.5 / π0-FAST は、openpi 互換のために差し替えた
        transformers（custom 4.53）を前提にしている。その版だけが siglip に
        check モジュールを持ち、素の transformers では ImportError になって
        「An incorrect transformer version is used」で構築が止まる
        （modeling_pi05.py:576-584）。

        一方で lerobot 0.4.4 の smolvla は transformers>=4.57.1 を要求しており、
        同じ環境に両方を満たす版は無い。差し替え版の配布先も 0.4.4 の
        メタデータには書かれていない（`pi` extra 自体が 0.4.4 に存在しない）。

        そこで PARC_PI_SKIP_TF_CHECK=1 のときだけ、確認を通すスタブを入れる。
        **これは検証を飛ばすのではなく、検証の方法を実測に移すという意味である。**
        差し替え版が SigLIP の実装そのものを変えているなら出力は壊れるが、
        それは公開 4 タスクの成功率を見れば一発で分かる（π0.5 は LIBERO で
        96.85% と報告されているので、壊れていれば 0 付近に出る）。

        重みが読めたかどうかは別問題で、pi05 の from_pretrained は例外を
        握り潰して**ランダム初期化のまま返す**経路を持つ。起動ログの
        「All keys loaded successfully!」を必ず確認すること。
        """
        import importlib
        import sys
        import types

        try:
            importlib.import_module("transformers.models.siglip.check")
            return
        except ImportError:
            pass

        if os.environ.get("PARC_PI_SKIP_TF_CHECK") != "1":
            raise RuntimeError(
                "この checkpoint (π0 系) は差し替え版 transformers を前提にしており、"
                " 素の transformers には transformers.models.siglip.check が無い。\n"
                "  PARC_PI_SKIP_TF_CHECK=1 を付けるとスタブで通せるが、"
                " SigLIP の実装差が出力に効く可能性があるため、\n"
                "  通したあとは必ず公開タスクで成功率を測ること。"
            )

        stub = types.ModuleType("transformers.models.siglip.check")
        stub.check_whether_transformers_replace_is_installed_correctly = lambda: True
        sys.modules["transformers.models.siglip.check"] = stub
        # `from transformers.models.siglip import check` は属性を先に見るので、
        # 親パッケージ側にも生やしておく。
        import transformers.models.siglip as _siglip

        _siglip.check = stub
        print(
            "[MyPolicy] 警告: transformers.models.siglip.check をスタブで通した"
            " (PARC_PI_SKIP_TF_CHECK=1)。\n"
            "[MyPolicy]   差し替え版 transformers を使っていないため、SigLIP の"
            " 実装差が出力に出る可能性がある。\n"
            "[MyPolicy]   公開タスクの成功率を測るまで提出してはいけない。"
        )

    def _resolve_image_keys(self) -> None:
        """モデルへ渡す画像キーを policy_preprocessor.json から導出する。

        rename_observations_processor が最初に走るため、こちらが渡すべきキーは
        rename_map の「左辺」である。追加学習でデータセットの feature 名が
        変わると右辺も左辺も変わるので、定数で持たず毎回読み直す。

        rename_map が空の checkpoint では、モデルの image feature 名を
        そのまま渡す形になるため、そちらを候補にする。

        main / wrist の割り当ては名前で判定し（WRIST_HINTS）、判定できない
        場合は右辺の cameraN の番号順、それも無ければ辞書順で先頭を main とする。
        導出に失敗した場合はクラス定数のフォールバック既定を使う。
        """
        import json
        import re

        prefix = "observation.images."
        try:
            cfg = json.loads((_WEIGHTS_DIR / "policy_preprocessor.json").read_text())
        except Exception as exc:
            print(f"[MyPolicy] 画像キーを導出できず既定を使う: {exc}")
            return

        rename: dict[str, str] = {}
        for step in cfg.get("steps", []):
            if step.get("registry_name") == "rename_observations_processor":
                rename = step.get("config", {}).get("rename_map", {}) or {}
                break

        if rename:
            # 左辺が渡すべきキー。右辺の cameraN の番号で並べる。
            def order(item):
                m = re.search(r"(\d+)$", item[1])
                return (int(m.group(1)) if m else 999, item[0])

            candidates = [k for k, _ in sorted(rename.items(), key=order)
                          if k.startswith(prefix)]
        else:
            # rename が無い場合はモデルの image feature 名をそのまま渡す。
            # empty_camera_* はプレースホルダなので候補から除く。
            feats = (cfg.get("steps") and self._normalizer_features(cfg)) or {}
            candidates = sorted(
                k for k, v in feats.items()
                if k.startswith(prefix)
                and "empty_camera" not in k
                and (v or {}).get("type") == "VISUAL"
            )

        if not candidates:
            print("[MyPolicy] 画像キーの候補が見つからず既定を使う。")
            return

        wrist = next(
            (k for k in candidates
             if any(h in k[len(prefix):].lower() for h in self.WRIST_HINTS)),
            None,
        )
        main = next((k for k in candidates if k != wrist), None)
        if wrist is None:
            main, wrist = candidates[0], (candidates[1] if len(candidates) > 1 else None)

        if main is None:
            print("[MyPolicy] main カメラを決められず既定を使う。")
            return

        self.key_main, self.key_wrist = main, wrist
        unused = [k for k in candidates if k not in (main, wrist)]
        if unused:
            print(f"[MyPolicy] 渡さない画像キー（観測が 2 つしか無いため）: {unused}")

    @staticmethod
    def _normalizer_features(cfg: dict) -> dict:
        """normalizer_processor の features 定義を取り出す。"""
        for step in cfg.get("steps", []):
            if step.get("registry_name") == "normalizer_processor":
                return step.get("config", {}).get("features", {}) or {}
        return {}

    def _preprocessor_overrides(self) -> dict:
        """保存済み preprocessor の、外部通信を要する設定を差し替える。

        vlm_model_name とは別に、policy_preprocessor.json の
        tokenizer_processor が独自に tokenizer_name を持っており、
        その値がハブ ID ("HuggingFaceTB/SmolVLM2-500M-Video-Instruct") のため
        AutoTokenizer.from_pretrained() が別経路でハブを見に行く。
        vlm_model_name だけ直しても採点環境では
        LocalEntryNotFoundError で起動に失敗する。

        あわせて device_processor の device も実機に合わせる
        （保存値は "cuda" 固定のため、GPU が無い環境で落ちる）。

        registry_name がそのまま override のキーになり、ユーザー指定が
        保存値に優先する（lerobot/processor/pipeline.py:731, 735）。
        存在しないキーを渡すと _validate_overrides_used が弾くため、
        実際に保存されているステップにだけ override を当てる。
        """
        import json

        try:
            cfg = json.loads((_WEIGHTS_DIR / "policy_preprocessor.json").read_text())
            names = {s.get("registry_name") for s in cfg.get("steps", [])}
        except Exception as exc:
            print(f"[MyPolicy] policy_preprocessor.json を読めず override を省略: {exc}")
            return {}

        overrides: dict = {}
        if "tokenizer_processor" in names and _BACKBONE_DIR.is_dir():
            overrides["tokenizer_processor"] = {"tokenizer_name": str(_BACKBONE_DIR)}
        if "device_processor" in names and self.device is not None:
            overrides["device_processor"] = {"device": str(self.device)}
        if overrides:
            print(f"[MyPolicy] preprocessor overrides: {overrides}")
        return overrides

    def _detect_state_dim(self, default: int = 8) -> int:
        """normalizer の統計から observation.state の次元を読む。

        config.json の input_features は [6] を宣言しているが、統計は 8 次元で
        あり、正規化は統計側の次元で行われる。実際に適用される側に合わせる。
        """
        try:
            for step in getattr(self.preprocessor, "steps", []):
                stats = getattr(step, "_tensor_stats", None)
                if stats and self.KEY_STATE in stats:
                    mean = stats[self.KEY_STATE].get("mean")
                    if mean is not None:
                        return int(mean.numel())
        except Exception as exc:
            print(f"[MyPolicy] state 次元の自動判定に失敗: {exc}")
        print(f"[MyPolicy] state 次元を判定できず、既定値 {default} を使う。")
        return default

    def _predict_chunk(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """観測から action chunk を推論し shape (N, 7) で返す。

        1 リクエストの制限は 10 秒。1 回でも超過するとトラック全体が 0 点になる。
        """
        if self.model is None:
            return np.zeros((self.ACTION_CHUNK_SIZE, 7), dtype=np.float32)

        if self.oft is not None:
            # OFT は uint8 HWC をそのまま受ける（180 度回転・224 へのリサイズ・
            # center crop は OFTModel 側で本家と同じ順序で行う）。
            return self.oft.predict_chunk(
                obs["agentview_image"],
                obs["robot0_eye_in_hand_image"],
                self._state_array(obs),
                self.instruction,
            )

        torch = self.torch
        batch = {
            self.key_main: self._to_image(obs["agentview_image"]),
            self.KEY_STATE: self._to_state(obs),
            "task": [self.instruction],
        }
        if self.key_wrist:
            batch[self.key_wrist] = self._to_image(obs["robot0_eye_in_hand_image"])
        dumping = (
            bool(_DEBUG_DIR) and not self._warming and self._dbg_n < _DEBUG_MAX
        )
        if dumping:
            self._dump_inputs(obs, batch)

        batch = self.preprocessor(batch)
        with torch.inference_mode():
            chunk = self.model.predict_action_chunk(batch)   # (1, chunk, 7)
        chunk = self.postprocessor(chunk)                    # 逆正規化

        arr = np.asarray(chunk.squeeze(0).float().cpu().numpy(), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 7:
            raise RuntimeError(f"予期しない chunk shape: {arr.shape}（期待 (N, 7)）")

        if dumping:
            self._dump_outputs(batch, arr)
            self._dbg_n += 1
        return arr

    # ------------------------------------------------------------
    # 切り分け用の計装。PARC_DEBUG_DIR を設定したときだけ動く。
    # 提出時の挙動には一切影響しない（環境変数が無ければ全て素通り）。
    # ------------------------------------------------------------

    def _dump_inputs(self, obs, batch) -> None:
        try:
            out = Path(_DEBUG_DIR)
            out.mkdir(parents=True, exist_ok=True)
            i = self._dbg_n

            from PIL import Image
            # 環境から届いた生画像
            Image.fromarray(obs["agentview_image"]).save(out / f"{i:02d}_raw_agentview.png")
            Image.fromarray(obs["robot0_eye_in_hand_image"]).save(out / f"{i:02d}_raw_wrist.png")
            # 実際にモデルへ入る画像（FLIP_IMAGES_180 適用後）
            fed = [("front", self.key_main)]
            if self.key_wrist:
                fed.append(("wrist", self.key_wrist))
            for name, key in fed:
                t = batch[key][0].permute(1, 2, 0).cpu().numpy()
                Image.fromarray((t * 255).clip(0, 255).astype(np.uint8)).save(
                    out / f"{i:02d}_fed_{name}.png"
                )

            state = batch[self.KEY_STATE][0].cpu().numpy()
            lines = [
                f"instruction = {self.instruction!r}",
                f"state (raw) = {np.round(state, 4).tolist()}",
            ]
            # 学習データの統計と突き合わせる（(x-mean)/std が ±3 を大きく超えたら分布外）
            stats = self._state_stats()
            if stats is not None:
                mean, std = stats
                z = (state - mean) / (std + 1e-8)
                lines += [
                    f"stats mean  = {np.round(mean, 4).tolist()}",
                    f"stats std   = {np.round(std, 4).tolist()}",
                    f"z-score     = {np.round(z, 2).tolist()}",
                    f"max|z|      = {float(np.max(np.abs(z))):.2f}"
                    "   ← 3 を大きく超えるなら state が学習分布の外",
                ]
            (out / f"{i:02d}_state.txt").write_text("\n".join(lines) + "\n")
            print(f"[MyPolicy][debug] {out}/{i:02d}_* を書き出した max|z| 判定は state.txt 参照")
        except Exception as exc:
            print(f"[MyPolicy][debug] 入力ダンプに失敗: {exc}")

    def _dump_outputs(self, processed_batch, chunk: np.ndarray) -> None:
        try:
            out = Path(_DEBUG_DIR)
            i = self._dbg_n
            lines = [
                f"chunk shape = {chunk.shape}",
                f"chunk[0]    = {np.round(chunk[0], 4).tolist()}",
                f"chunk[-1]   = {np.round(chunk[-1], 4).tolist()}",
                f"per-dim min = {np.round(chunk.min(axis=0), 4).tolist()}",
                f"per-dim max = {np.round(chunk.max(axis=0), 4).tolist()}",
                f"per-dim std = {np.round(chunk.std(axis=0), 4).tolist()}",
                f"|xyz| 平均  = {float(np.mean(np.abs(chunk[:, :3]))):.4f}"
                "   ← 0.01 未満ならほぼ静止指令",
                f"gripper     = {np.round(chunk[:, 6], 3).tolist()}",
            ]
            (out / f"{i:02d}_action.txt").write_text("\n".join(lines) + "\n")
        except Exception as exc:
            print(f"[MyPolicy][debug] 出力ダンプに失敗: {exc}")

    def _trace_step(self, obs: dict[str, np.ndarray]) -> None:
        """毎ステップ、手先と各物体の距離を CSV に追記する。

        観測には <物体名>_to_robot0_eef_pos が含まれる（評価側の rollout.py が
        衝突判定でこのサフィックスを除外していることから存在が分かる）。
        そのノルムが手先と物体の距離になる。

        画像の向きが正しければ、対象物体との距離はエピソード中に単調に
        近づくはずである。向きが誤っていれば無関係に動き回る。
        FLIP_IMAGES_180 を True / False で振って min 距離を比べれば、
        目視に頼らず客観的に判定できる。
        """
        try:
            keys = sorted(k for k in obs if k.endswith("_to_robot0_eef_pos"))
            path = Path(_DEBUG_DIR) / "trace.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            if self._trace_i == 0:
                cols = ["step", "eef_x", "eef_y", "eef_z"]
                cols += [k[: -len("_to_robot0_eef_pos")] for k in keys]
                path.write_text(",".join(cols) + "\n")
                print(f"[MyPolicy][debug] trace 対象の物体: "
                      f"{[k[: -len('_to_robot0_eef_pos')] for k in keys]}")
            eef = np.asarray(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float64)
            row = [str(self._trace_i)] + [f"{v:.4f}" for v in eef]
            row += [f"{float(np.linalg.norm(np.asarray(obs[k], dtype=np.float64))):.4f}"
                    for k in keys]
            with path.open("a") as f:
                f.write(",".join(row) + "\n")
            self._trace_i += 1
        except Exception as exc:
            print(f"[MyPolicy][debug] trace に失敗: {exc}")
            self._trace_i += 1

    def _state_stats(self):
        """normalizer が持つ observation.state の mean / std を numpy で返す。"""
        try:
            for step in getattr(self.preprocessor, "steps", []):
                stats = getattr(step, "_tensor_stats", None)
                if stats and self.KEY_STATE in stats:
                    s = stats[self.KEY_STATE]
                    return (s["mean"].float().cpu().numpy(),
                            s["std"].float().cpu().numpy())
        except Exception:
            pass
        return None

    def _to_image(self, hwc_uint8: np.ndarray):
        """uint8 HWC -> float32 (1, 3, H, W) [0, 1]、必要なら 180 度回転。

        512 へのリサイズと [-1, 1] への変換は SmolVLAPolicy.prepare_images が
        内部で行うため、ここでは行わない（二重適用になる）。
        """
        torch = self.torch
        t = torch.from_numpy(np.ascontiguousarray(hwc_uint8))
        t = t.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        if self.FLIP_IMAGES_180:
            t = torch.flip(t, dims=[2, 3])
        return t.to(self.device)

    def _to_state(self, obs: dict[str, np.ndarray]):
        """eef_pos(3) + axis_angle(3) [+ gripper_qpos(2)] を組み立てる。

        state_dim が 8 を超える checkpoint（π0 系は max_state_dim=32 を宣言し、
        統計もその次元で保存されていることがある）では後ろをゼロで埋める。
        lerobot の pad_vector と同じ扱いで、正規化は統計の次元で走るため、
        こちらが短い配列を渡すと形が合わずに落ちる。
        """
        state = self._state_array(obs)
        return self.torch.from_numpy(state).unsqueeze(0).to(self.device)

    def _state_array(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """observation.state を numpy で組み立てる（torch を挟まない経路用）。"""
        parts = [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(3),
            self._quat2axisangle(obs["robot0_eef_quat"]),
        ]
        if self.state_dim >= 8:
            parts.append(
                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(2)
            )
        state = np.concatenate(parts).astype(np.float32)
        if state.size < self.state_dim:
            state = np.pad(state, (0, self.state_dim - state.size))
        return state

    @staticmethod
    def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
        """クォータニオン (x, y, z, w) を軸角 (3,) へ変換する。

        lerobot/processor/env_processor.py:113-152 の _quat2axisangle と
        同じ式・同じゼロ割ガード (den > 1e-10) を numpy に写したもの。
        """
        q = np.asarray(quat, dtype=np.float32).reshape(4)
        w = float(np.clip(q[3], -1.0, 1.0))
        den = float(np.sqrt(max(0.0, 1.0 - w * w)))
        if den <= 1e-10:
            return np.zeros(3, dtype=np.float32)
        return ((q[:3] / den) * (2.0 * np.arccos(w))).astype(np.float32)

    # ------------------------------------------------------------
    # 以下は原則そのままで動く（提出物としての作法）
    # ------------------------------------------------------------

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        t0 = time.perf_counter()
        if _DEBUG_DIR and not self._warming:
            self._trace_step(obs)
        if self.TEMPORAL_ENSEMBLE:
            action = self._sanitize(self._ensembled_action(obs))
        else:
            if not self._queue:
                chunk = self._fresh_chunk(obs)
                self._queue.extend(chunk[: max(1, self.N_ACTION_EXEC)])
            action = self._sanitize(self._queue.popleft())
        if not self._warming:
            self._record_latency(time.perf_counter() - t0)
        return action

    def _fresh_chunk(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """_predict_chunk() を呼び、shape (N, 7) float32 として検証して返す。"""
        chunk = np.asarray(self._predict_chunk(obs), dtype=np.float32).reshape(-1, 7)
        if chunk.shape[0] == 0:
            raise RuntimeError("_predict_chunk() が空の action を返した。")
        return chunk

    def _ensembled_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """ACT の temporal ensembling で 1 ステップぶんの action を作る。

        いま実行するステップ t に対して、過去の推論が出した予測が複数ある
        （t で推論したチャンクの先頭、t-1 で推論したチャンクの 2 番目、…）。
        これらはすべて「ステップ t で取るべき action」の推定値なので、
        平均すれば推論ごとのサンプリングノイズが落ちる。チャンク境界で
        予測を切り替える代わりに混ぜる、というのがこの手法である。

        重みは ACT と同じ w_i = exp(-m * i)（i=0 が最も古い予測）。
        """
        if self._step % self.ENSEMBLE_QUERY_EVERY == 0 or not self._ens:
            chunk = self._fresh_chunk(obs)
            self._ens.append([0, chunk[: max(1, self.ENSEMBLE_HORIZON)]])
        self._step += 1

        # 古い順に、各予測が「今のステップ」に対して出している action を集める
        preds = np.stack([entry[1][entry[0]] for entry in self._ens]).astype(np.float64)
        w = np.exp(-self.ENSEMBLE_M * np.arange(len(preds), dtype=np.float64))
        action = (preds * (w / w.sum())[:, None]).sum(axis=0)
        if not self.ENSEMBLE_GRIPPER:
            action[6] = preds[-1][6]      # 最新の予測をそのまま使う

        for entry in self._ens:
            entry[0] += 1
        while self._ens and self._ens[0][0] >= len(self._ens[0][1]):
            self._ens.popleft()
        return action

    def _ensemble_desc(self) -> str:
        """起動ログ用。どの設定で走っているかをサーバーログに残す。"""
        if not self.TEMPORAL_ENSEMBLE:
            return f"off (exec={self.N_ACTION_EXEC})"
        return (
            f"on h={self.ENSEMBLE_HORIZON} query={self.ENSEMBLE_QUERY_EVERY}"
            f" m={self.ENSEMBLE_M:g} gripper={'avg' if self.ENSEMBLE_GRIPPER else 'latest'}"
            " (exec は使わない)"
        )

    def _clear_episode_state(self) -> None:
        """エピソード境界で捨てる状態。持ち越すと前エピソードの action が
        次エピソードの冒頭に流れ込み、衝突の原因になる。"""
        self._queue.clear()
        self._ens.clear()
        self._step = 0

    def _record_latency(self, dt: float) -> None:
        """/act の所要時間を記録し、遅い応答をその場で報告する。

        評価側は 1 リクエスト 10 秒で打ち切り、超えるとトラック全体が
        0 点になる。平均ではなく最悪値が効くので、外れ値を必ず残す。
        """
        self._lat_n += 1
        self._lat_sum += dt
        if dt > self._lat_max:
            self._lat_max = dt
        if dt > self.SLOW_REQUEST_SEC:
            self._lat_slow += 1
            print(
                f"[MyPolicy] 遅い /act: {dt:.2f}s"
                f" (これまでの最大 {self._lat_max:.2f}s"
                f" / {self._lat_n} リクエスト目 / 遅延 {self._lat_slow} 回)"
                " ★10 秒でトラックが 0 点になる",
                flush=True,
            )

    def reset(self, instruction: str = "") -> None:
        if self._lat_n:
            print(
                f"[MyPolicy] 前エピソードのレイテンシ: n={self._lat_n}"
                f" mean={self._lat_sum / self._lat_n:.3f}s max={self._lat_max:.3f}s"
                f" slow(>{self.SLOW_REQUEST_SEC:g}s)={self._lat_slow}",
                flush=True,
            )
        self._lat_max = self._lat_sum = 0.0
        self._lat_n = self._lat_slow = 0

        self.instruction = instruction
        self._clear_episode_state()
        if self.model is not None and hasattr(self.model, "reset"):
            self.model.reset()   # SmolVLA 内部の action queue もクリアする

    @staticmethod
    def _sanitize(action: np.ndarray) -> np.ndarray:
        """評価側に返す前の最終防衛。shape (7,) float32、NaN/Inf 無し、[-1, 1]。"""
        a = np.asarray(action, dtype=np.float32).reshape(7)
        a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def _warmup(self) -> None:
        """ダミー観測で推論を空打ちし、初回のみ発生する遅延を起動時に吸収する。

        CUDA カーネルのコンパイル等は初回推論で走る。それが 1 エピソード目の
        1 ステップ目に起きると 10 秒制限に触れうるため、120 秒の枠がある
        起動時に済ませておく。エピソード状態を都度捨てて実推論を 2 回強制する
        （ensembling が有効で ENSEMBLE_QUERY_EVERY > 1 のときも、捨てないと
        2 回目が既存の予測から作られてしまい実推論にならない）。
        """
        dummy = {
            "agentview_image": np.zeros((128, 128, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((128, 128, 3), dtype=np.uint8),
            "robot0_joint_pos": np.zeros(7, dtype=np.float64),
            "robot0_eef_pos": np.zeros(3, dtype=np.float64),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            "robot0_gripper_qpos": np.zeros(2, dtype=np.float64),
        }
        self._warming = True   # warmup のダミー観測はデバッグダンプの対象外
        try:
            for _ in range(2):
                self._clear_episode_state()
                self.get_action(dummy)
        except Exception as exc:
            print(f"[MyPolicy] warmup に失敗（無視して続行）: {exc}")
        finally:
            self._warming = False
            self.reset("")


# ============================================================
# 以下は変更不可
# ============================================================


def deserialize_obs(data: bytes) -> dict[str, np.ndarray]:
    unpacked = msgpack.unpackb(data, raw=False)
    obs = {}
    for key, val in unpacked.items():
        arr = np.frombuffer(val["data"], dtype=np.dtype(val["dtype"]))
        obs[key] = arr.reshape(val["shape"]).copy()
    return obs


def serialize_action(action: np.ndarray) -> bytes:
    return msgpack.packb(
        {"data": action.astype(np.float32).tobytes()},
        use_bin_type=True,
    )


app = FastAPI(title="VLA Policy Server")
_policy: BasePolicy | None = None


def set_policy(policy: BasePolicy) -> None:
    global _policy
    _policy = policy


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
async def reset_policy(request: Request):
    body = await request.body()
    instruction = ""
    if body:
        import json
        data = json.loads(body)
        instruction = data.get("instruction", "")
    _policy.reset(instruction=instruction)
    return {"status": "ok"}


@app.post("/act")
async def act(request: Request):
    body = await request.body()
    obs = deserialize_obs(body)
    action = _policy.get_action(obs)
    return Response(
        content=serialize_action(action),
        media_type="application/x-msgpack",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    set_policy(MyPolicy())
    print(f"Policy server starting on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
