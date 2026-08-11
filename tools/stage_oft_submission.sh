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

# 作り直す前に必ず消す。前回の残骸が混ざった zip は、中身を見ないと
# 気づけないまま提出されうる。
#
# rm -rf が「ディレクトリは空ではありません」で失敗することがある。
# NFS 上でそのファイルを開いているプロセスがあると、削除の代わりに
# .nfsXXXX へ silly-rename されて残るためである（サーバーがまだ重みを
# 掴んでいる、zip が走っている等）。原因を出さずに続けると、古い
# policy_server.py が入った zip ができる。
rm -rf "$DST" 2>/dev/null || true
if [ -e "$DST" ]; then
    echo "ERROR: $DST を消せない。中身を掴んでいるプロセスがある。" >&2
    echo "--- 残っているもの ---" >&2
    find "$DST" -maxdepth 2 | head -20 >&2
    echo "--- 掴んでいるプロセス（fuser があれば）---" >&2
    command -v fuser >/dev/null 2>&1 && fuser -vm "$DST" 2>&1 | head -10 >&2
    echo "  ポリシーサーバーが動いていないか確認すること: pkill -f policy_server.py" >&2
    exit 1
fi
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
REQ="$DST/requirements.txt"
cat > "$REQ" <<'EOF'
# OpenVLA-OFT+ (LIBERO-plus mix-SFT) を動かすための依存。
# lerobot は使わないので同梱していない。
#
# verify_clean_env.sh はまっさらな venv にこのファイルだけを入れて検証する。
# つまりここに書いていないものは採点環境に存在しない。torch も含めて全部要る。
#
# transformers は checkpoint の保存時と同じ版に固定する（config.json の
# transformers_version = 4.40.1）。4.5x では PreTrainedModel まわりが変わって
# おり、trust_remote_code の modeling_prismatic.py が読めない。
#
# torch / torchvision / numpy / pillow は SmolVLA 版と同じ版に固定する。
# あちらは実際に採点を通っており、採点環境でその wheel が取れることが
# 分かっている。加えて parc-oft の torch も 2.10.0 で、そこで OFT が動く
# ことを実測済みである。解決を pip に任せるより、両方で確認できている
# 組み合わせを指定するほうが安全である。
torch==2.10.0
torchvision==0.25.0
numpy==2.2.6
pillow==12.3.0
transformers==4.40.1
tokenizers==0.19.1
timm==0.9.10
accelerate
safetensors
huggingface_hub
sentencepiece
protobuf
fastapi>=0.68
uvicorn>=0.15
# policy_server.py はモジュール先頭で msgpack を import し、/act の本体を
# msgpack.unpackb で復号する。これが無いとサーバーは起動すらしない。
msgpack
requests
EOF

# SmolVLA 版の requirements があるなら、サーバー本体（fastapi / uvicorn）と
# torch の指定を突き合わせる。あちらは実際に採点を通っている構成なので、
# 版指定が食い違っていたら気づけるようにしておく。
if [ -f submission/requirements.txt ]; then
    echo
    echo "[stage] --- SmolVLA 版との指定差（左: OFT / 右: SmolVLA）---"
    for pkg in torch torchvision fastapi uvicorn numpy pillow; do
        a="$(grep -iE "^${pkg}([=<>!~[]|$)" "$REQ" | head -1)"
        b="$(grep -iE "^${pkg}([=<>!~[]|$)" submission/requirements.txt | head -1)"
        printf '  %-12s %-24s %s\n' "$pkg" "${a:--}" "${b:--}"
    done
fi

# --- 構成の宣言 --------------------------------------------------------------
# 採点環境では環境変数を渡せないので、提出ごとの構成はこのファイルで持つ。
# policy_server.py が起動時に読み、既に設定済みの環境変数は上書きしない。
# コードを書き換えずに構成を振れるので、どの提出が何だったかが zip の中身を
# 見るだけで分かる。
if [ -n "${PARC_STAGE_ENV:-}" ]; then
    printf '%s\n' ${PARC_STAGE_ENV//,/ } > "$DST/parc_env"
    echo "[stage] parc_env:"
    sed 's/^/    /' "$DST/parc_env"
else
    echo "[stage] parc_env なし（policy_server.py の既定＝本家構成で走る）"
fi

echo
echo "[stage] --- 構成 ---"
du -sh "$DST"
du -sh "$DST"/* | sort -rh
echo
echo "[stage] 次:"
echo "  PARC_SUBMISSION_SRC=$DST bash tools/make_submission.sh --dry-run"
