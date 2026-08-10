#!/usr/bin/env python
"""LoRA adapter をベースモデルへマージし、評価できる形で書き出す。

    python tools/merge_lora.py runs/lora_all40_r8
    python tools/merge_lora.py runs/lora_all40aug_r8 --out runs/merged_aug

マージ後のディレクトリは policy_server.py がそのまま読める形にする
（config + 重み + preprocessor / postprocessor）。評価は重みを
入れ替えずに環境変数で切り替える。

    PARC_WEIGHTS_DIR=$PWD/runs/merged_all40_r8 bash tools/run_policy_server.sh

学習時に --policy.empty_cameras=0 を渡しており、ベースの config（2）とは
値が変わる。推論側もマージ後の config を読むので学習時と揃うが、
差分は必ず表示して目視できるようにしてある。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path


def find_policy_python() -> str:
    """parc-policy の python を探す（run_policy_server.sh と同じ方針）。

    エラーメッセージにそのまま貼れるコマンドを作るためだけに使う。
    見つからなければプレースホルダを返す。
    """
    env = os.environ.get("PARC_POLICY_PYTHON")
    if env:
        return env
    for base in ("miniforge3", "miniconda3", "anaconda3"):
        candidate = Path.home() / base / "envs" / "parc-policy" / "bin" / "python"
        if candidate.is_file():
            return str(candidate)
    return "/path/to/envs/parc-policy/bin/python"

# preprocessor / postprocessor は policy.save_pretrained() の対象外なので
# 学習チェックポイント（無ければベース）から持ってくる。
PROCESSOR_PATTERNS = (
    "policy_preprocessor.json",
    "policy_preprocessor*.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor*.safetensors",
)

# 差分を確認したい config の項目。empty_cameras は学習時に 0 を渡している。
CONFIG_KEYS_OF_INTEREST = (
    "empty_cameras",
    "n_action_steps",
    "chunk_size",
    "n_obs_steps",
    "resize_imgs_with_padding",
    "vlm_model_name",
    "use_peft",
)


def find_checkpoint(run_dir: Path, step: int | None) -> Path:
    root = run_dir / "checkpoints"
    if not root.is_dir():
        raise SystemExit(f"checkpoints が無い: {root}")

    if step is not None:
        candidate = root / f"{step:06d}" / "pretrained_model"
        if not candidate.is_dir():
            raise SystemExit(f"見つからない: {candidate}")
        return candidate

    steps = sorted(p for p in root.iterdir() if p.name.isdigit())
    if not steps:
        raise SystemExit(f"checkpoint が 1 つも無い: {root}")
    return steps[-1] / "pretrained_model"


def copy_processors(dest: Path, checkpoint: Path, base: Path) -> None:
    for pattern in PROCESSOR_PATTERNS:
        # チェックポイント側を優先し、無ければベースから拾う。
        sources = sorted(checkpoint.glob(pattern)) or sorted(base.glob(pattern))
        if not sources:
            print(f"  警告: {pattern} が checkpoint にもベースにも無い")
            continue
        for src in sources:
            shutil.copy2(src, dest / src.name)
            print(f"  processor: {src.name}  <- {src.parent.name}")


def summarize_config(path: Path) -> dict:
    config_path = path / "config.json"
    if not config_path.is_file():
        return {}
    data = json.loads(config_path.read_text())
    return {k: data.get(k) for k in CONFIG_KEYS_OF_INTEREST if k in data}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="lerobot-train の --output_dir")
    parser.add_argument("--step", type=int, help="使う checkpoint。既定は最終")
    parser.add_argument(
        "--base",
        default="submission/model_weights",
        help="マージ先のベースモデル（既定: submission/model_weights）",
    )
    parser.add_argument("--out", help="出力先。既定は runs/merged_<run_dir の名前>")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    base = Path(args.base).resolve()
    checkpoint = find_checkpoint(run_dir, args.step)
    out = Path(args.out).resolve() if args.out else run_dir.parent / f"merged_{run_dir.name}"

    if not (checkpoint / "adapter_model.safetensors").is_file():
        raise SystemExit(f"adapter が無い: {checkpoint}/adapter_model.safetensors")
    if not base.is_dir():
        raise SystemExit(f"ベースモデルが無い: {base}")

    print(f"checkpoint : {checkpoint}")
    print(f"base       : {base}")
    print(f"out        : {out}\n")

    # torch / lerobot / peft は parc-policy にしか入っていない。conda base の
    # python で叩くと ModuleNotFoundError になる。素の traceback だと原因が
    # 「インタプリタを間違えた」ことだと分かりにくいので、直せる形で落とす。
    try:
        import torch

        # lerobot 0.4.4 の lerobot.configs は名前空間パッケージで PreTrainedConfig を
        # 再エクスポートしない（`from lerobot.configs import ...` は unknown location で
        # 失敗する）。policy_server.py:262 と同じ場所から取る。
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from peft import PeftModel
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"{exc.name} が見つからない。parc-policy の python で実行すること:\n"
            f"    {find_policy_python()} {' '.join(sys.argv)}"
        ) from exc

    config = PreTrainedConfig.from_pretrained(str(checkpoint), local_files_only=True)
    config.device = "cpu"
    config.pretrained_path = str(base)
    config.use_peft = False

    print("ベースを読み込み中...")
    policy = SmolVLAPolicy.from_pretrained(
        str(base), config=config, strict=False, local_files_only=True
    )

    print("adapter を適用してマージ中...")
    peft_policy = PeftModel.from_pretrained(
        policy, str(checkpoint), is_trainable=False, torch_device="cpu"
    )
    merged = peft_policy.merge_and_unload(safe_merge=True)

    # 保存した config が学習・推論以外の状態を持ち回らないようにする。
    merged.config.use_peft = False
    merged.config.pretrained_path = None
    merged.config.push_to_hub = False
    merged.config.repo_id = None
    merged.config.device = None

    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out))
    print(f"\n保存: {out}")

    copy_processors(out, checkpoint, base)

    del merged, peft_policy, policy
    gc.collect()
    torch.cuda.empty_cache()

    base_cfg = summarize_config(base)
    out_cfg = summarize_config(out)
    print("\n=== config の差分（ベース -> マージ後）===")
    for key in sorted(set(base_cfg) | set(out_cfg)):
        before, after = base_cfg.get(key), out_cfg.get(key)
        mark = "  " if before == after else "* "
        print(f"{mark}{key}: {before!r} -> {after!r}")
    print("\n* の付いた項目は推論時の挙動が変わりうる。")
    print("policy_server.py は vlm_model_name を同梱バックボーンで上書きするので、")
    print("その行が変わっていても実害は無い。")

    print("\n評価するには:")
    print(f"    PARC_WEIGHTS_DIR={out} bash tools/run_policy_server.sh")


if __name__ == "__main__":
    main()
