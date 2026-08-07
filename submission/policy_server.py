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
    #: 学習時の設定は 50（config の n_action_steps）だが、実測では
    #: 下げるほど良い。put_the_bowl_on_the_stove_light_11 を各 10 エピソード:
    #:     n_exec  success  collision  cartesian  jerk(rms)  sparc
    #:        25     0.60      0.20      1.475      8.82    -3.51
    #:        10     0.80      0.20      0.823      6.89    -2.32
    #: 成功率だけでなく経路長・ジャーク・SPARC も同方向に改善しており、
    #: 軌跡が短く滑らかになっている。50 は 2 エピソードで 0.00 だった。
    #:
    #: 推論は 1 回 0.24 秒、制限は 10 秒なので回数を増やす余地は大きい。
    #: 600 ステップのエピソードでの推論回数と所要:
    #:     50 -> 12 回 / 2.9 秒、25 -> 24 回 / 5.8 秒、10 -> 60 回 / 14 秒
    #:
    #: A/B 用に PARC_N_EXEC で上書きできる。既定（未設定）は 10。
    N_ACTION_EXEC = _env_int("PARC_N_EXEC", 10)

    def __init__(self):
        self.instruction = ""
        self._queue: deque[np.ndarray] = deque()
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
            policy_cfg=cfg,
            pretrained_path=str(_WEIGHTS_DIR),
            preprocessor_overrides=self._preprocessor_overrides(),
        )

        self._resolve_image_keys()
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
            f"\n[MyPolicy]   main={self.key_main} wrist={self.key_wrist}"
        )
        return model

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
        if _DEBUG_DIR and not self._warming:
            self._trace_step(obs)
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
        self._warming = True   # warmup のダミー観測はデバッグダンプの対象外
        try:
            for _ in range(2):
                self._queue.clear()
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
