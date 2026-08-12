#!/usr/bin/env bash
#
# 既存の提出 zip から、parc_env だけ差し替えた別構成の zip を作る。
#
#   bash tools/respin_parc_env.sh submission_oft_tta4.zip submission_oft_tta4m.zip \
#        "PARC_ENSEMBLE=1,PARC_ENS_H=8,PARC_ENS_GRIPPER=0,PARC_OFT_TTA=4,PARC_ENS_M=-0.1"
#
# 構成違いの提出は parc_env の 1 行しか違わない。それを作るのに 15 GB を
# 読み直して 12 GB を書き直すのは無駄で、しかもその間 NFS が詰まって
# 同時に走らせているスイープの起動待ちを潰す（実際に条件が 1 つ飛んだ）。
#
# zip はアーカイブ内の 1 エントリだけを置き換えられる。コピー（ローカル
# ディスクなら 1〜2 分）＋ 数 KB の書き換えで済む。
#
# 出来上がった zip は必ず verify_clean_env.sh に通すこと。parc_env の綴りを
# 間違えても静かに既定で走るだけなので、起動ログでしか気づけない。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SRC="${1:-}"
DST="${2:-}"
ENVSPEC="${3:-}"
if [ -z "$SRC" ] || [ -z "$DST" ] || [ -z "$ENVSPEC" ]; then
    echo 'usage: bash tools/respin_parc_env.sh <src.zip> <dst.zip> "KEY=VAL,KEY=VAL"' >&2
    exit 2
fi
[ -f "$SRC" ] || { echo "ERROR: $SRC が無い" >&2; exit 1; }
[ "$SRC" = "$DST" ] && { echo "ERROR: 元と同じ名前にはできない" >&2; exit 1; }

command -v zip >/dev/null 2>&1 || { echo "ERROR: zip が無い" >&2; exit 1; }

echo "[respin] $SRC -> $DST"
cp -f "$SRC" "$DST"

# zip は「カレントからの相対パス」でエントリ名を決める。parc_env を zip 直下に
# 置きたいので、一時ディレクトリの中で作業する。
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
printf '%s\n' ${ENVSPEC//,/ } > "$TMP/parc_env"

echo "[respin] parc_env:"
sed 's/^/    /' "$TMP/parc_env"

( cd "$TMP" && zip -q "$OLDPWD/$DST" parc_env )

echo
echo "[respin] --- zip 内の parc_env ---"
unzip -p "$DST" parc_env | sed 's/^/    /'
echo "[respin] サイズ: $(du -h "$DST" | cut -f1)"
echo
echo "[respin] 次: PARC_ZIP=$DST bash tools/verify_clean_env.sh"
echo "         （起動ログの [OFT] ... tta= とレイテンシで効いたか確認する）"
