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


class MyPolicy(BasePolicy):
    """ベースラインポリシー。

    モデル未搭載でも評価パイプラインを最後まで通せる状態にしてある。
    モデルを組み込むときに触るのは次の 2 箇所だけでよい。

        _load_model()    : 起動時に 1 回だけ呼ばれる。重みのロードをここに書く
        _predict_chunk() : 観測から action を推論する。ここに推論を書く

    get_action() / reset() 側には、提出物として必要な作法
    （出力の shape・dtype 正規化、NaN/Inf 除去、値域クリップ、
    エピソード境界でのキャッシュ破棄）を既に入れてある。

    現状の _predict_chunk() はゼロ action を返すため、成功率は 0 になる。
    ランダム action ではなく「静止」を既定にしているのは、
    ランダムだと周囲の物体に接触して collision_rate が汚れ、
    モデル組み込み後の比較対象として使えなくなるためである。
    """

    #: 1 回の推論で生成する action の本数（action chunking を使わないなら 1）
    ACTION_CHUNK_SIZE = 1

    #: 生成したチャンクのうち、実際に環境へ流すステップ数。
    #: ACTION_CHUNK_SIZE より小さくすると推論頻度が上がり閉ループ性が増す。
    N_ACTION_EXEC = 1

    def __init__(self):
        self.instruction = ""
        self._queue: deque[np.ndarray] = deque()
        self.model = self._load_model()
        self._warmup()

    # ------------------------------------------------------------
    # モデルを組み込むときに編集する箇所
    # ------------------------------------------------------------

    def _load_model(self):
        """重みをロードして返す。モデル未搭載なら None を返す。

        サーバー起動から /health が 200 を返すまでの制限は 120 秒である。
        重い初期化はすべてここで済ませ、get_action() には持ち込まないこと。

        実装例:
            import torch
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            model = SmolVLAPolicy.from_pretrained(_WEIGHTS_DIR)
            return model.eval().to(self.device)
        """
        if not _WEIGHTS_DIR.is_dir():
            print(
                f"[MyPolicy] {_WEIGHTS_DIR} が無いため、ゼロ action の"
                " ベースラインで動作する（成功率は 0 になる）。"
            )
            return None

        # TODO: ここで重みをロードする
        print(
            f"[MyPolicy] {_WEIGHTS_DIR} を検出したが _load_model() が未実装のため、"
            " ゼロ action のベースラインで動作する。"
        )
        return None

    def _predict_chunk(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """観測から action を推論し、shape (N, 7) で返す。

        1 リクエストの制限は 10 秒である。1 回でも超過するとトラック全体が
        0 点になるため、ここが唯一の重い処理であることを意識すること。

        利用できる観測キー（BasePolicy.get_action の docstring 参照）:
            agentview_image           (128, 128, 3) uint8
            robot0_eye_in_hand_image  (128, 128, 3) uint8
            robot0_joint_pos          (7,)  float
            robot0_eef_pos            (3,)  float
            robot0_eef_quat           (4,)  float
            robot0_gripper_qpos       (2,)  float

        言語指示は self.instruction に入っている。

        実装例（action chunking）:
            batch = self._preprocess(obs)
            with torch.no_grad():
                chunk = self.model.predict_action_chunk(batch)
            return chunk.squeeze(0).cpu().numpy()
        """
        if self.model is None:
            # ベースライン: 何もしない（静止）
            return np.zeros((self.ACTION_CHUNK_SIZE, 7), dtype=np.float32)

        # TODO: ここで推論する
        return np.zeros((self.ACTION_CHUNK_SIZE, 7), dtype=np.float32)

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
        起動時に済ませておく。
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
