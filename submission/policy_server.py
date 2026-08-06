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
_WEIGHTS_DIR = _HERE / "model_weights"
_BACKBONE_DIR = _HERE / "smolvlm_backbone"


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

    # --- 提出物の観測 -> SmolVLA の入力キー（rename_map の左辺）---------------
    KEY_MAIN = "observation.images.front"    # agentview      -> camera1
    KEY_WRIST = "observation.images.wrist"   # eye_in_hand    -> camera2
    KEY_STATE = "observation.state"

    #: LIBERO の生画像を LeRobot 側の向きへ揃える 180 度回転。
    #: LiberoProcessorStep (lerobot/processor/env_processor.py) が
    #: torch.flip(img, dims=[2, 3]) で行っているのと同じ変換である。
    #: state のレイアウトが同 Step の出力と一致することから同系統の変換と判断した。
    #: 成功率が 0 のまま動かない場合は、まずここを False にして A/B すること。
    FLIP_IMAGES_180 = True

    #: config.json の chunk_size / n_action_steps に一致させる
    ACTION_CHUNK_SIZE = 50

    #: 生成したチャンクのうち実際に環境へ流すステップ数。
    #: 小さくすると推論頻度が上がり閉ループ性が増す（衝突対策の調整点）。
    N_ACTION_EXEC = 50

    def __init__(self):
        self.instruction = ""
        self._queue: deque[np.ndarray] = deque()
        self.torch = None
        self.device = None
        self.preprocessor = None
        self.postprocessor = None
        self.state_dim = 8
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
            )
            return None

        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.torch = torch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        cfg = PreTrainedConfig.from_pretrained(str(_WEIGHTS_DIR), local_files_only=True)

        # 採点環境は外部通信が無く HF キャッシュも存在しない。
        # バックボーンを同梱している場合はローカルパスを見るよう差し替える。
        if _BACKBONE_DIR.is_dir():
            cfg.vlm_model_name = str(_BACKBONE_DIR)
            print(f"[MyPolicy] VLM backbone: {_BACKBONE_DIR}")
        else:
            print(
                f"[MyPolicy] 警告: {_BACKBONE_DIR} が無い。"
                f" vlm_model_name={getattr(cfg, 'vlm_model_name', '?')} を"
                " HF キャッシュから解決するため、採点環境では起動に失敗する。"
            )

        model = SmolVLAPolicy.from_pretrained(
            str(_WEIGHTS_DIR), config=cfg, local_files_only=True
        )
        model = model.eval().to(self.device)

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=str(_WEIGHTS_DIR)
        )

        self.state_dim = self._detect_state_dim()
        if self.state_dim not in (6, 8):
            raise RuntimeError(
                f"observation.state の次元 {self.state_dim} に対応する構成が不明。"
                " 6 (eef_pos+axis_angle) か 8 (+gripper_qpos) のみ対応する。"
            )

        print(
            f"[MyPolicy] SmolVLA ready | device={self.device}"
            f" | state_dim={self.state_dim} | chunk={self.ACTION_CHUNK_SIZE}"
            f" | exec={self.N_ACTION_EXEC} | flip180={self.FLIP_IMAGES_180}"
        )
        return model

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

        torch = self.torch
        batch = {
            self.KEY_MAIN: self._to_image(obs["agentview_image"]),
            self.KEY_WRIST: self._to_image(obs["robot0_eye_in_hand_image"]),
            self.KEY_STATE: self._to_state(obs),
            "task": [self.instruction],
        }
        batch = self.preprocessor(batch)
        with torch.inference_mode():
            chunk = self.model.predict_action_chunk(batch)   # (1, chunk, 7)
        chunk = self.postprocessor(chunk)                    # 逆正規化

        arr = np.asarray(chunk.squeeze(0).float().cpu().numpy(), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 7:
            raise RuntimeError(f"予期しない chunk shape: {arr.shape}（期待 (N, 7)）")
        return arr

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
        """eef_pos(3) + axis_angle(3) [+ gripper_qpos(2)] を組み立てる。"""
        torch = self.torch
        parts = [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(3),
            self._quat2axisangle(obs["robot0_eef_quat"]),
        ]
        if self.state_dim == 8:
            parts.append(
                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(2)
            )
        state = np.concatenate(parts).astype(np.float32)
        return torch.from_numpy(state).unsqueeze(0).to(self.device)

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
        if not self._queue:
            chunk = np.asarray(self._predict_chunk(obs), dtype=np.float32)
            chunk = chunk.reshape(-1, 7)
            if chunk.shape[0] == 0:
                raise RuntimeError("_predict_chunk() が空の action を返した。")
            self._queue.extend(chunk[: max(1, self.N_ACTION_EXEC)])
        return self._sanitize(self._queue.popleft())

    def reset(self, instruction: str = "") -> None:
        # エピソード境界。チャンクのキャッシュを持ち越すと、前エピソードの
        # action が次エピソードの冒頭に流れ込んで衝突の原因になる。
        self.instruction = instruction
        self._queue.clear()
        if self.model is not None:
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
        起動時に済ませておく。キューを都度捨てて実推論を 2 回強制する。
        """
        dummy = {
            "agentview_image": np.zeros((128, 128, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((128, 128, 3), dtype=np.uint8),
            "robot0_joint_pos": np.zeros(7, dtype=np.float64),
            "robot0_eef_pos": np.zeros(3, dtype=np.float64),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            "robot0_gripper_qpos": np.zeros(2, dtype=np.float64),
        }
        try:
            for _ in range(2):
                self._queue.clear()
                self.get_action(dummy)
        except Exception as exc:
            print(f"[MyPolicy] warmup に失敗（無視して続行）: {exc}")
        finally:
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
