"""OpenVLA-OFT+ (LIBERO-plus mix-SFT) の推論ラッパー。

なぜ SmolVLA から乗り換えるのか
------------------------------
LIBERO-Plus のリーダーボード（論文 arXiv:2510.13626 / モデルカード）では

    OpenVLA-OFT+  79.6   (Camera 92.8)
    OpenVLA-OFT   70.0   (Camera 56.4)
    π0-Fast       64.2   (Camera 65.1)
    π0            54.6   (Camera 13.8)
    OpenVLA       17.3   (Camera  0.8)

で、SmolVLA は評価対象にすら入っていない。採点の失敗モードは「8 本中 6 本が
300 step でゴールに届かない」であって軌道の滑らかさではないので、効く可能性が
あるのはモデルの汎化力そのものだけである。OFT+ は LIBERO-plus データで
mix-SFT されており、全モデルの急所であるカメラ視点摂動で 92.8 と突出している。

lerobot は使わない
------------------
OFT は transformers==4.40.1 ピンで、lerobot 0.4.4 が SmolVLA 用に要求する
transformers>=4.57.1 と同居できない。ただし OFT 側は lerobot を一切必要と
しないので、環境ごと分けれられる（tools/setup_oft_env.sh）。

checkpoint に何が入っているか
----------------------------
    model-0000{1..4}-of-00004.safetensors   本体 15.1 GB (bf16)
    modeling_prismatic.py / configuration_prismatic.py / processing_prismatic.py
                                            trust_remote_code のコード。同梱
                                            されているのでオフラインで載る
    action_head--150000_checkpoint.pt       L1 回帰ヘッド (bf16, 302 MB)
    proprio_projector--150000_checkpoint.pt 固有受容感覚の射影 (fp32, 67 MB)
    dataset_statistics.json                 action / proprio の正規化統計

`modeling_prismatic.py` は `prismatic.vla.constants` と
`prismatic.training.train_utils` を import する。openvla-oft 本体は入れず、
その 2 つだけを vendor_oft/ に写してある。

推論の手順は openvla-oft の experiments/robot/openvla_utils.py の
get_vla_action と、libero/run_libero_eval.py の GenerateConfig の既定
（use_l1_regression=True / use_film=False / num_images_in_input=2 /
use_proprio=True / center_crop=True / num_open_loop_steps=8）に合わせてある。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_VENDOR_OFT = _HERE / "vendor_oft"

# LIBERO 系の固定値。vendor_oft 側と同じ値だが、こちらは import せずに使える
# ようにしておく（sys.path を触る前に参照する箇所があるため）。
ACTION_DIM = 7
PROPRIO_DIM = 8
NUM_ACTIONS_CHUNK = 8
LLM_DIM = 4096
IMAGE_SIZE = 224
CENTER_CROP_SCALE = 0.9


def _install_vendor_path() -> None:
    """同梱の prismatic シムを import 可能にする。

    checkpoint の modeling_prismatic.py が `from prismatic.vla.constants import ...`
    を行うため、本物の openvla-oft が入っていない環境ではこれが無いと
    ModuleNotFoundError になる。
    """
    import sys

    p = str(_VENDOR_OFT)
    if _VENDOR_OFT.is_dir() and p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# ヘッド（openvla-oft の prismatic/models/action_heads.py, projectors.py の移植）
#
# 形は checkpoint の state_dict と突き合わせて確認済み:
#   action head      layer_norm1(28672) / fc1(4096,28672) / blocks x2 / fc2(7,4096)
#   proprio projector fc1(4096,8) / fc2(4096,4096)
# ---------------------------------------------------------------------------


def _build_head_classes():
    import torch.nn as nn

    class MLPResNetBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim
            self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())

        def forward(self, x):
            return x + self.ffn(x)

    class MLPResNet(nn.Module):
        def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.layer_norm1 = nn.LayerNorm(input_dim)
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.mlp_resnet_blocks = nn.ModuleList(
                [MLPResNetBlock(dim=hidden_dim) for _ in range(num_blocks)]
            )
            self.layer_norm2 = nn.LayerNorm(hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = self.relu(self.fc1(self.layer_norm1(x)))
            for block in self.mlp_resnet_blocks:
                x = block(x)
            return self.fc2(self.layer_norm2(x))

    class L1RegressionActionHead(nn.Module):
        def __init__(self, input_dim=LLM_DIM, hidden_dim=LLM_DIM, action_dim=ACTION_DIM):
            super().__init__()
            self.action_dim = action_dim
            self.model = MLPResNet(
                num_blocks=2,
                input_dim=input_dim * ACTION_DIM,
                hidden_dim=hidden_dim,
                output_dim=action_dim,
            )

        def predict_action(self, actions_hidden_states):
            bsz = actions_hidden_states.shape[0]
            rearranged = actions_hidden_states.reshape(bsz, NUM_ACTIONS_CHUNK, -1)
            return self.model(rearranged)

    class ProprioProjector(nn.Module):
        def __init__(self, llm_dim=LLM_DIM, proprio_dim=PROPRIO_DIM):
            super().__init__()
            self.llm_dim = llm_dim
            self.proprio_dim = proprio_dim
            self.fc1 = nn.Linear(proprio_dim, llm_dim, bias=True)
            self.fc2 = nn.Linear(llm_dim, llm_dim, bias=True)
            self.act_fn1 = nn.GELU()

        def forward(self, proprio=None):
            return self.fc2(self.act_fn1(self.fc1(proprio)))

    return L1RegressionActionHead, ProprioProjector


def _load_checkpoint_into(module, path: Path, torch):
    """`module.` 接頭辞（DDP で保存されたもの）を外して strict にロードする。

    strict にするのは意図的である。π0.5 では lerobot 側が例外を握り潰して
    ランダム初期化のまま返し、それに気づかず評価しかけた。黙って壊れるより
    起動時に落ちるほうがよい。
    """
    sd = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    module.load_state_dict(sd, strict=True)
    return module


# ---------------------------------------------------------------------------
# 画像処理（openvla-oft の prepare_images_for_vla と同じ手順）
# ---------------------------------------------------------------------------


def _resize_lanczos(hwc_uint8: np.ndarray, size: int) -> np.ndarray:
    """Lanczos3 で size x size へ。

    本家は tf.image.resize(method="lanczos3", antialias=True) を使う。
    TensorFlow を入れたくないので PIL の LANCZOS（同じ a=3 の Lanczos で、
    縮小時は内部で antialias 相当の処理をする）で置き換えている。
    """
    from PIL import Image

    if hwc_uint8.shape[0] == size and hwc_uint8.shape[1] == size:
        return hwc_uint8
    img = Image.fromarray(np.ascontiguousarray(hwc_uint8)).convert("RGB")
    return np.asarray(img.resize((size, size), Image.LANCZOS), dtype=np.uint8)


def _center_crop(hwc_uint8: np.ndarray, torch, crop_scale: float = CENTER_CROP_SCALE) -> np.ndarray:
    """本家の crop_and_resize と同じ、中央 sqrt(crop_scale) を切って戻す。

    tf.image.crop_and_resize は正規化座標 n を画素 n*(H-1) に写し、box の
    両端を出力の両端に合わせる。これは grid_sample の align_corners=True と
    同じ対応なので、そちらで書き直してある（F.interpolate では box の端が
    半画素ずれる）。
    """
    h, w = hwc_uint8.shape[:2]
    side = float(np.sqrt(crop_scale))
    lo, hi = (1.0 - side) / 2.0, (1.0 - side) / 2.0 + side

    # PIL 由来の配列は read-only なので copy してから渡す（torch が警告を出す）
    x = torch.from_numpy(np.ascontiguousarray(hwc_uint8).copy())
    x = x.permute(2, 0, 1)[None].float() / 255.0

    # 正規化座標 -> 画素 -> grid_sample の [-1, 1]
    ys = torch.linspace(lo * (h - 1), hi * (h - 1), h) * (2.0 / (h - 1)) - 1.0
    xs = torch.linspace(lo * (w - 1), hi * (w - 1), w) * (2.0 / (w - 1)) - 1.0
    grid = torch.stack(torch.meshgrid(ys, xs, indexing="ij")[::-1], dim=-1)[None]

    out = torch.nn.functional.grid_sample(
        x, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    out = out.clamp(0.0, 1.0)[0].permute(1, 2, 0).numpy()
    # tf.image.convert_image_dtype(..., saturate=True) は *255 + 0.5 の切り捨て
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def prepare_image(
    hwc_uint8: np.ndarray, torch, flip180: bool, center_crop, crop_scale: float | None = None
) -> np.ndarray:
    """PARC から届く uint8 HWC を OFT の入力形式へ。

    crop_scale を渡すとその値で切る（TTA 用）。省略時は既定の 0.9。
    center_crop が偽ならクロップしない。
    """
    img = hwc_uint8
    if flip180:
        img = img[::-1, ::-1]
    img = _resize_lanczos(np.ascontiguousarray(img), IMAGE_SIZE)
    if center_crop:
        img = _center_crop(img, torch, crop_scale or CENTER_CROP_SCALE)
    return img


#: TTA で使うクロップ倍率。既定の 0.9 を中心に前後へ振る。
#: OFT は学習時にランダムクロップを使っているので、この範囲は分布内である。
TTA_CROP_SCALES = (0.90, 0.95, 0.85, 1.00)


def process_gripper(chunk: np.ndarray, binarize: bool = True) -> np.ndarray:
    """gripper 次元をモデル出力から環境の規約へ直す。

    本家 run_libero_eval は env.step の直前で process_action() を通しており、
    それが 2 段階になっている（experiments/robot/robot_utils.py）。

      1. normalize_gripper_action: RLDS のデータローダは gripper だけ [0, 1] に
         標準化しているので、他の次元と同じ [-1, +1] へ戻す。
         y = 2x - 1。binarize=True なら sign を取る
      2. invert_gripper_action: そのデータローダは 0=閉じる / 1=開く で
         揃えているが、環境は -1=開く / +1=閉じる なので符号を反転する

    これを入れないと**グリッパーが常に逆に動く**。実測でもモデル出力の
    gripper 次元は 1.04 付近（統計は q01=0.0 / q99=1.0）で、そのまま渡すと
    clip されて +1（＝閉じる）になり、開くべき場面で閉じる。
    """
    out = np.array(chunk, dtype=np.float32, copy=True)
    g = 2.0 * out[..., -1] - 1.0
    if binarize:
        g = np.sign(g)
    out[..., -1] = -g
    return out


def normalize_proprio(state: np.ndarray, stats: dict) -> np.ndarray:
    """[q01, q99] -> [-1, 1] にして clip する（本家 normalize_proprio と同一）。"""
    low = np.asarray(stats["q01"], dtype=np.float64)
    high = np.asarray(stats["q99"], dtype=np.float64)
    mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    normalized = np.clip(2.0 * (state - low) / (high - low + 1e-8) - 1.0, -1.0, 1.0)
    return np.where(mask, normalized, state)


# ---------------------------------------------------------------------------
# 本体
# ---------------------------------------------------------------------------


class OFTModel:
    """OpenVLA-OFT+ を読み込み、観測から action chunk (8, 7) を返す。"""

    PROMPT = "In: What action should the robot take to {task}?\nOut:"

    def __init__(
        self,
        weights_dir: Path,
        *,
        device: str | None = None,
        flip180: bool = True,
        center_crop: bool = True,
        unnorm_key: str | None = None,
        gripper_transform: bool = True,
        gripper_binarize: bool = True,
        tta_views: int = 1,
    ) -> None:
        _install_vendor_path()

        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.torch = torch
        self.weights_dir = Path(weights_dir)
        self.flip180 = flip180
        self.center_crop = center_crop
        self.gripper_transform = gripper_transform
        self.gripper_binarize = gripper_binarize
        self.tta_views = max(1, min(int(tta_views), len(TTA_CROP_SCALES)))
        self.device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )

        print(f"[OFT] weights: {self.weights_dir}")
        self.processor = self._timed(
            "processor",
            lambda: AutoProcessor.from_pretrained(
                str(self.weights_dir), trust_remote_code=True, local_files_only=True
            ),
        )

        # サーバー起動から /health が 200 を返すまでの制限は 120 秒である。
        # 15 GB を CPU 上に組んでから GPU へコピーすると、その往復だけで
        # 予算を使い切る。device_map で shard を直接 GPU へ流すと CPU 常駐と
        # コピーが消える。accelerate が要るが parc-oft には入っている。
        common = dict(
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True,
        )
        use_map = self.device.type == "cuda" and env_flag("PARC_OFT_DEVICE_MAP", True)
        if use_map:
            try:
                self.vla = self._timed(
                    "model (device_map)",
                    lambda: AutoModelForVision2Seq.from_pretrained(
                        str(self.weights_dir),
                        device_map={"": self.device.index or 0},
                        **common,
                    ),
                )
            except Exception as exc:
                print(f"[OFT] device_map で載らなかったので CPU 経由に落とす: {exc}")
                use_map = False
        if not use_map:
            self.vla = self._timed(
                "model (cpu->gpu)",
                lambda: AutoModelForVision2Seq.from_pretrained(
                    str(self.weights_dir), **common
                ).to(self.device),
            )
        self.vla = self.vla.eval()

        # 画像 2 枚（主カメラ + 手首）を受けられるようにする。
        if hasattr(self.vla, "vision_backbone") and hasattr(
            self.vla.vision_backbone, "set_num_images_in_input"
        ):
            self.vla.vision_backbone.set_num_images_in_input(2)
        else:
            raise RuntimeError(
                "vision_backbone.set_num_images_in_input が無い。"
                " OFT 版ではない checkpoint を指している可能性がある。"
            )

        # 正規化統計。config.json にも埋まっているが、fine-tune が出した
        # dataset_statistics.json のほうを本家 get_vla と同じく優先する。
        stats_path = self.weights_dir / "dataset_statistics.json"
        if stats_path.is_file():
            self.vla.norm_stats = json.loads(stats_path.read_text())
        self.norm_stats = self.vla.norm_stats

        self.unnorm_key = self._resolve_unnorm_key(unnorm_key)

        L1RegressionActionHead, ProprioProjector = _build_head_classes()
        self.action_head = self._timed(
            "action_head",
            lambda: _load_checkpoint_into(
                L1RegressionActionHead(), self._find_ckpt("action_head"), torch
            ).to(self.device, dtype=torch.bfloat16).eval(),
        )
        self.proprio_projector = self._timed(
            "proprio_projector",
            lambda: _load_checkpoint_into(
                ProprioProjector(), self._find_ckpt("proprio_projector"), torch
            ).to(self.device, dtype=torch.bfloat16).eval(),
        )

        grip = "off"
        if self.gripper_transform:
            grip = "binarize" if self.gripper_binarize else "linear"
        print(
            f"[OFT] ready | device={self.device} | unnorm_key={self.unnorm_key}"
            f" | chunk={NUM_ACTIONS_CHUNK} | flip180={self.flip180}"
            f" | center_crop={self.center_crop} | gripper={grip}"
            f" | tta={self.tta_views}"
        )

    # -- 準備 ---------------------------------------------------------------

    @staticmethod
    def _timed(label: str, fn):
        """段階ごとの所要時間を出す。120 秒の起動制限に対する内訳が要る。"""
        import time

        t = time.perf_counter()
        out = fn()
        print(f"[OFT] {label}: {time.perf_counter() - t:.1f}s")
        return out

    def _find_ckpt(self, prefix: str) -> Path:
        matches = sorted(self.weights_dir.glob(f"{prefix}--*_checkpoint.pt"))
        if not matches:
            raise FileNotFoundError(
                f"{self.weights_dir} に {prefix}--*_checkpoint.pt が無い。"
                " OFT のヘッドが揃っていない checkpoint である。"
            )
        if len(matches) > 1:
            print(f"[OFT] {prefix} の候補が複数ある。最後のものを使う: {matches[-1].name}")
        return matches[-1]

    def _resolve_unnorm_key(self, requested: str | None) -> str:
        keys = list(self.norm_stats)
        if requested:
            if requested not in keys:
                raise KeyError(f"unnorm_key={requested} が統計に無い。候補: {keys}")
            return requested
        libero = [k for k in keys if k.startswith("libero")]
        if not libero:
            raise KeyError(f"libero の統計が見つからない。候補: {keys}")
        # この checkpoint では 4 スイートの統計が完全に一致しているので、
        # どれを選んでも同じ値になる。そのことを起動時に確かめておく。
        first = self.norm_stats[libero[0]]["action"]
        for k in libero[1:]:
            other = self.norm_stats[k]["action"]
            if other["q01"] != first["q01"] or other["q99"] != first["q99"]:
                print(
                    f"[OFT] 警告: {libero[0]} と {k} で action の統計が異なる。"
                    " 採点タスクスイートは非公開なので、選択が結果に効く。"
                )
                break
        return sorted(libero)[0]

    # -- 推論 ---------------------------------------------------------------

    def predict_chunk(
        self,
        image_main: np.ndarray,
        image_wrist: np.ndarray | None,
        state: np.ndarray,
        instruction: str,
    ) -> np.ndarray:
        """観測から action chunk を推論して shape (8, 7) を返す。

        tta_views > 1 のときはクロップ倍率を変えて複数回推論し、平均を取る。

        temporal ensembling が時間方向の平均であるのに対し、これは空間方向の
        平均である。L1 回帰ヘッドの OFT は出力が決定的なので、同じ画像を
        何度推論しても同じ値しか出ない。入力側を振らないと平均する意味が無い。

        平均は gripper 変換の**前**に取る。変換後は ±1 に二値化されており、
        平均すると中間値になって二値化が壊れる。変換前の gripper は [0, 1] の
        連続値なので、そこで平均してから sign を取れば多数決になる。
        """
        raw = np.stack(
            [
                self._predict_raw(image_main, image_wrist, state, instruction, scale)
                for scale in TTA_CROP_SCALES[: self.tta_views]
            ]
        ).mean(axis=0)

        chunk = raw.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM).astype(np.float32)
        if self.gripper_transform:
            chunk = process_gripper(chunk, binarize=self.gripper_binarize)
        return chunk

    def _predict_raw(
        self,
        image_main: np.ndarray,
        image_wrist: np.ndarray | None,
        state: np.ndarray,
        instruction: str,
        crop_scale: float,
    ) -> np.ndarray:
        """1 視点ぶんの生の chunk（gripper 変換前）を返す。"""
        torch = self.torch

        primary = prepare_image(
            image_main, torch, self.flip180, self.center_crop, crop_scale
        )
        prompt = self.PROMPT.format(task=(instruction or "").lower().strip())

        inputs = self.processor(prompt, self._to_pil(primary)).to(
            self.device, dtype=torch.bfloat16
        )
        if image_wrist is not None:
            wrist = prepare_image(
                image_wrist, torch, self.flip180, self.center_crop, crop_scale
            )
            wrist_inputs = self.processor(prompt, self._to_pil(wrist)).to(
                self.device, dtype=torch.bfloat16
            )
            inputs["pixel_values"] = torch.cat(
                [inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1
            )

        proprio = normalize_proprio(
            np.asarray(state, dtype=np.float64).reshape(-1),
            self.norm_stats[self.unnorm_key]["proprio"],
        )

        with torch.inference_mode():
            out = self.vla.predict_action(
                **inputs,
                unnorm_key=self.unnorm_key,
                do_sample=False,
                proprio=proprio,
                proprio_projector=self.proprio_projector,
                action_head=self.action_head,
                use_film=False,
            )
        # 戻り値は (unnormalized_actions, actions_hidden_states) のタプルである
        # （型注釈は np.ndarray と書いてあるが実際は 2 要素。
        #  modeling_prismatic.py:1055）。隠れ状態は GPU 上のテンソルなので、
        # タプルのまま numpy へ渡すと変換に失敗する。
        actions = out[0] if isinstance(out, tuple) else out
        if hasattr(actions, "detach"):
            actions = actions.detach().float().cpu().numpy()
        return np.asarray(actions, dtype=np.float32).reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

    @staticmethod
    def _to_pil(hwc_uint8: np.ndarray):
        from PIL import Image

        return Image.fromarray(np.ascontiguousarray(hwc_uint8)).convert("RGB")


def is_oft_checkpoint(path: Path) -> bool:
    """weights ディレクトリが OpenVLA 系かどうかを config.json から判定する。"""
    cfg = Path(path) / "config.json"
    if not cfg.is_file():
        return False
    try:
        return json.loads(cfg.read_text()).get("model_type") == "openvla"
    except Exception:
        return False


def env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v not in ("0", "false", "False", "")
