"""document.xml から描画高さを見積もる。

LibreOffice がこのコンテナで動かないため、ページ数を目視確認できない。
代わりに文字幅を全角/半角で分けて行数を数え、twip で高さを積む。
細かい禁則処理は無視しているので、±15% 程度の精度と考えること。
"""
import re
import sys
import zipfile

A4_H, A4_W = 16838, 11906

NS = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'


#: 半角文字の幅（em 比）。Yu Gothic の Latin は小文字で 0.5em 前後だが、
#: 大文字・数字は 0.6em を超える。**多めに見て安全側へ倒す** — 見積りが
#: 実寸を下回ると 2 ページ制限を静かに割る側に間違えることになる。
ASCII_EM = 0.58


def char_w(ch, half_pt):
    """1 文字の幅を twip で返す。全角は font size ぶん、半角はその ASCII_EM 倍。"""
    full_pt = half_pt / 2.0
    if ord(ch) < 0x2000 or ch in '−·→×':
        return full_pt * 20 * ASCII_EM
    return full_pt * 20


def line_count(text, half_pt, width):
    if not text:
        return 1
    w = sum(char_w(c, half_pt) for c in text)
    return max(1, int(w / width) + (1 if w % width else 0))


PAGE_H = 0
BODY_W = 0


def main(path):
    with zipfile.ZipFile(path) as z:
        xml = z.read('word/document.xml').decode('utf-8')

    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml)
    body = root.find(f'{NS}body')

    # 余白は sectPr から読む。ここを決め打ちにすると build 側で余白を変えた
    # ときに黙って古い容量で割り続ける。
    mar = body.find(f'{NS}sectPr/{NS}pgMar')
    top = int(mar.get(f'{NS}top', 850)) if mar is not None else 850
    bot = int(mar.get(f'{NS}bottom', 700)) if mar is not None else 700
    left = int(mar.get(f'{NS}left', 850)) if mar is not None else 850
    right = int(mar.get(f'{NS}right', 850)) if mar is not None else 850
    global PAGE_H, BODY_W
    PAGE_H = A4_H - top - bot
    BODY_W = A4_W - left - right

    total = 0.0
    detail = []

    def para_h(pel, width, default_sz=17):
        """段落 1 つの高さ（行 + 前後の空き）。"""
        txt = ''.join(t.text or '' for t in pel.iter(f'{NS}t'))
        sz = default_sz
        szel = pel.find(f'.//{NS}sz')
        if szel is not None:
            sz = int(szel.get(f'{NS}val'))
        spacing = pel.find(f'{NS}pPr/{NS}spacing')
        before = after = 0
        line = 240
        rule = 'auto'
        if spacing is not None:
            before = int(spacing.get(f'{NS}before', 0) or 0)
            after = int(spacing.get(f'{NS}after', 0) or 0)
            line = int(spacing.get(f'{NS}line', 240) or 240)
            rule = spacing.get(f'{NS}lineRule', 'auto')
        if rule == 'exact':
            # 行高が twip で確定する。見積りの不確かさが折り返し数だけになる。
            lh = line
        else:
            # auto は 240 分率。フォントの hhea 由来で 1.2〜1.6em と幅があり、
            # Yu Gothic は特に大きい。ここを踏むと見積りが当てにならない。
            lh = (sz / 2.0) * 20 * 1.45 * (line / 240.0)
        n = line_count(txt, sz, width)
        return n * lh + before + after, txt[:40], n

    for child in body:
        tag = child.tag
        if tag == f'{NS}p':
            h, t, n = para_h(child, BODY_W)
            total += h
            if t.strip():
                detail.append((round(h), n, t))
        elif tag == f'{NS}tbl':
            grid = [int(g.get(f'{NS}w')) for g in child.findall(f'{NS}tblGrid/{NS}gridCol')]
            for row in child.findall(f'{NS}tr'):
                rh = 0
                for i, tc in enumerate(row.findall(f'{NS}tc')):
                    w = grid[i] if i < len(grid) else BODY_W
                    w -= 160  # セル左右マージン
                    ch = sum(para_h(pp, w)[0] for pp in tc.findall(f'{NS}p'))
                    rh = max(rh, ch)
                rh += 80  # セル上下マージン
                total += rh
                detail.append((round(rh), '-', '[表の行]'))
        elif tag == f'{NS}sectPr':
            pass

    print(f'本文高さ見積り : {total:,.0f} twip')
    print(f'1 ページの容量 : {PAGE_H:,} twip')
    print(f'推定ページ数   : {total / PAGE_H:.2f} ページ')
    print(f'2 ページに対する余裕: {(1 - total / (PAGE_H * 2)) * 100:.0f}%')
    print()
    print('高さ上位:')
    for h, n, t in sorted(detail, reverse=True)[:8]:
        print(f'  {h:>5} twip  {n:>3} 行  {t}')


if __name__ == '__main__':
    main(sys.argv[1])
