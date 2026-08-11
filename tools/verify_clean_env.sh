#!/usr/bin/env bash
#
# 採点環境と同条件で提出物を検証する。これを通さずに提出しない。
#
#   bash tools/verify_clean_env.sh                # submission.zip を検証
#   bash tools/verify_clean_env.sh --keep         # 検証用 venv を残す
#
# 採点環境の再現ポイント:
#   1. まっさらな venv に requirements.txt だけを入れる
#      （参加者の作業環境に既に入っているパッケージに依存していないか）
#   2. --only-binary=:all: で wheel のみ許可
#      採点環境には Python.h (python3-dev) が無く、C 拡張のソースビルドは
#      必ず失敗する。実際に evdev のビルド失敗で 0 点になった。
#      wheel のみに制限すれば同じ失敗をローカルで先に踏める。
#   3. HF キャッシュを隠す（採点環境にユーザーキャッシュは無い）
#   4. その venv の python で validate_submission.py を動かす
#      同スクリプトはサーバーを sys.executable で起動するため、
#      サーバーもこのクリーン環境で立ち上がる
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ZIP="${PARC_ZIP:-submission.zip}"
KEEP=0
[ "${1:-}" = "--keep" ] && KEEP=1
test -f "$ZIP" || { echo "ERROR: $ZIP が無い。先に bash tools/make_submission.sh" >&2; exit 1; }

PY="${PARC_POLICY_PYTHON:-}"
if [ -z "$PY" ]; then
    if command -v conda >/dev/null 2>&1; then
        base="$(conda info --base 2>/dev/null || true)"
        [ -n "$base" ] && PY="$base/envs/parc-policy/bin/python"
    fi
    [ -x "${PY:-}" ] || PY="$HOME/miniforge3/envs/parc-policy/bin/python"
fi
[ -x "$PY" ] || { echo "ERROR: python が見つからない。PARC_POLICY_PYTHON を指定すること。" >&2; exit 1; }

VENV="${PARC_VERIFY_VENV:-/tmp/parc_verify_venv}"
HFDIR="/tmp/parc_verify_empty_hf"
cleanup() { [ "$KEEP" = "1" ] || rm -rf "$VENV"; }
trap cleanup EXIT

echo "=============================================="
echo " クリーン環境での提出物検証"
echo "   zip    : $ZIP ($(du -h "$ZIP" | cut -f1))"
echo "   base py: $PY ($("$PY" -V 2>&1))"
echo "   venv   : $VENV"
echo "=============================================="

rm -rf "$VENV" "$HFDIR"; mkdir -p "$HFDIR"
"$PY" -m venv "$VENV" || { echo "ERROR: venv を作成できない" >&2; exit 1; }
VPY="$VENV/bin/python"
"$VPY" -m pip install -q --upgrade pip >/dev/null 2>&1

TMPREQ="$(mktemp)"
unzip -p "$ZIP" requirements.txt > "$TMPREQ" || { echo "ERROR: zip に requirements.txt が無い" >&2; exit 1; }

echo
echo "--- 1. requirements を wheel のみで導入（採点環境に Python.h は無い）---"
if ! "$VPY" -m pip install --only-binary=:all: -r "$TMPREQ"; then
    echo
    echo "★ 失敗。wheel だけでは導入できない。"
    echo "  どれに wheel が無いかを調べる..."
    while read -r line; do
        pkg="${line%%#*}"; pkg="$(echo "$pkg" | xargs)"
        [ -z "$pkg" ] && continue
        if ! "$VPY" -m pip download --only-binary=:all: --no-deps \
                -d /tmp/parc_wheelprobe "$pkg" >/dev/null 2>&1; then
            echo "    wheel なし: $pkg"
        fi
    done < "$TMPREQ"
    rm -rf /tmp/parc_wheelprobe
    echo
    echo "  ここに出ないのに失敗する場合は、依存の依存に wheel が無い"
    echo "  （エラー本文の 'No matching distribution found for X' を見ること）。"
    echo
    echo "  対処の順番:"
    echo "    1. そのパッケージが本当に必要か確認する。不要なら"
    echo "       tools/vendor_lerobot.sh の PKGS から外す"
    echo "    2. 必要な場合、それが純 Python の sdist なら採点環境でも"
    echo "       ビルドできる（Python.h が要るのは C 拡張だけ）。"
    echo "       その判断で通すなら、この検証はここで止まる点に注意すること"
    rm -f "$TMPREQ"; exit 1
fi
rm -f "$TMPREQ"
echo "--- wheel のみで導入できた ---"

echo
echo "--- 2. validate_submission（HF キャッシュを隠して起動まで確認）---"
# 動的スモーク（サーバーを起動して /health /reset /act を叩く）は
# クライアント側に numpy / msgpack / requests を要求し、無ければ黙って
# スキップして PASS を出す。それでは「起動を確認した」ことにならない。
#
# しかも危険な向きに壊れる。提出物の requirements から msgpack が抜けていると、
# 同じ venv を使うスモークが道具不足でスキップされ、**msgpack が要るという
# 事実を検出すべきテストが、msgpack が無いせいで動かない**。
# 実際に OFT 版の初回ビルドがこれで PASS した（policy_server.py は
# モジュール先頭で msgpack を import するので、採点では 0 点になっていた）。
#
# 検証用の道具として venv に入れておく。提出物の requirements とは別物で、
# 提出物側に msgpack が要るかどうかは、この下のスキップ検出が判定する。
"$VPY" -m pip install -q --only-binary=:all: msgpack requests >/dev/null 2>&1 || true

SMOKELOG="$(mktemp)"
HF_HOME="$HFDIR" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$VPY" validate_submission.py "$ZIP" 2>&1 | tee "$SMOKELOG"
rc="${PIPESTATUS[0]}"

if grep -q 'smoke.deps_missing\|smoke.skipped' "$SMOKELOG"; then
    echo
    echo "★ 動的スモークがスキップされた。サーバーの起動は確認できていない。"
    echo "  この検証は PASS にしない。"
    rc=1
fi
rm -f "$SMOKELOG"

echo
if [ "$rc" = "0" ]; then
    echo "=============================================="
    echo " PASS: 採点環境と同条件で起動・応答を確認した"
    echo "=============================================="
else
    echo "=============================================="
    echo " FAIL: この状態で提出しないこと (rc=$rc)"
    echo "=============================================="
fi
[ "$KEEP" = "1" ] && echo "venv を残した: $VENV"
exit $rc
