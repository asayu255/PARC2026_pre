#!/usr/bin/env bash
#
# OpenVLA-OFT+ の提出物を submission_oft/ に組み立てる。
#
#   PARC_OFT_WEIGHTS=~/parc_models/oft_libero_plus bash tools/stage_oft_submission.sh
#   PARC_SUBMISSION_SRC=submission_oft bash tools/make_submission.sh
#
# SmolVLA 版（submission/）とは中身が別物なので、上書きせず別ディレクトリに置く。
# 提出を切り替えても元に戻せるようにするためで、最良採用である以上、
# 戻せることには実際の価値がある。
#
# lerobot の同梱は要らない。あれは pynput -> evdev が C 拡張で、採点環境に
# Python.h が無くビルドできないため pip 経路を避けたものだった（§1）。
# OFT が要るのは transformers / tokenizers / timm / accelerate だけで、
# どれも wheel がある。checkpoint 側に modeling_prismatic.py が同梱されて
# いるので trust_remote_code もオフラインで解決する。
#
# 重みは 15 GB あるのでハードリンクで置く（同一ファイルシステムなら一瞬で、
# 追加の容量も食わない）。zip はハードリンクを普通のファイルとして格納する。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SRC_W="${PARC_OFT_WEIGHTS:-$HOME/parc_models/oft_libero_plus}"
DST="${PARC_OFT_STAGE:-submission_oft}"

SRC_W="${SRC_W/#\~/$HOME}"
[ -d "$SRC_W" ] || { echo "ERROR: $SRC_W が無い" >&2; exit 1; }
[ -f "$SRC_W/config.json" ] || { echo "ERROR: $SRC_W/config.json が無い" >&2; exit 1; }

echo "[stage] weights: $SRC_W"
echo "[stage] dest   : $DST"

rm -rf "$DST"
mkdir -p "$DST/model_weights"

# --- サーバー本体 ------------------------------------------------------------
cp submission/policy_server.py "$DST/"
cp submission/oft_policy.py    "$DST/"
cp -r submission/vendor_oft    "$DST/"
find "$DST/vendor_oft" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

# --- 重み --------------------------------------------------------------------
# lora_adapter/ は要らない。4 シャードは既に mix-SFT 済みの重みで、
# アダプタを重ねて使うものではない（README の配布形態）。
# .cache/ は huggingface_hub のダウンロード管理用。
copied=0
for f in "$SRC_W"/*; do
    name="$(basename "$f")"
    case "$name" in
        lora_adapter|.cache|.gitattributes|README.md) continue ;;
    esac
    if ! cp -al "$f" "$DST/model_weights/$name" 2>/dev/null; then
        cp -r "$f" "$DST/model_weights/$name"      # 別 FS ならコピーに落とす
    fi
    copied=$((copied + 1))
done
echo "[stage] model_weights に $copied 件"

# 必須ファイルの確認。どれが欠けてもオフラインでは起動できない。
for need in config.json model.safetensors.index.json dataset_statistics.json \
            modeling_prismatic.py configuration_prismatic.py processing_prismatic.py \
            preprocessor_config.json processor_config.json tokenizer.json; do
    [ -e "$DST/model_weights/$need" ] || { echo "ERROR: $need が無い" >&2; exit 1; }
done
for prefix in action_head proprio_projector; do
    ls "$DST/model_weights/$prefix"--*_checkpoint.pt >/dev/null 2>&1 \
        || { echo "ERROR: $prefix--*_checkpoint.pt が無い" >&2; exit 1; }
done

# --- requirements ------------------------------------------------------------
# transformers は checkpoint が保存された版に合わせて固定する
# （config.json の transformers_version = 4.40.1）。それ以外は採点環境の
# 解決に任せる。torch を固定しないのは、採点機の CUDA に合う版を向こうで
# 選ばせるためで、こちらが指定して外すほうが危ない。
cat > "$DST/requirements.txt" <<'EOF'
# OpenVLA-OFT+ (LIBERO-plus mix-SFT) を動かすための依存。
# lerobot は使わないので同梱していない。
#
# transformers は checkpoint の保存時と同じ版に固定する。4.5x では
# PreTrainedModel まわりが変わっており、trust_remote_code の
# modeling_prismatic.py が読めない。
transformers==4.40.1
tokenizers==0.19.1
timm==0.9.10
accelerate
safetensors
huggingface_hub
sentencepiece
protobuf
numpy
pillow
fastapi
uvicorn
EOF

echo
echo "[stage] --- 構成 ---"
du -sh "$DST"
du -sh "$DST"/* | sort -rh
echo
echo "[stage] 次:"
echo "  PARC_SUBMISSION_SRC=$DST bash tools/make_submission.sh --dry-run"
