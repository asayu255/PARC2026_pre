const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
} = require('docx');
const fs = require('fs');

const FONT = 'Yu Gothic';
const W = 9840;            // usable width (DXA) for ~15mm margins on A4
const GREY = 'F2F2F2';
const RULE = { style: BorderStyle.SINGLE, size: 4, color: 'AAAAAA' };

function txt(s, o = {}) {
  return new TextRun({ text: s, font: FONT, size: o.size || 17, bold: !!o.bold, italics: !!o.it, color: o.color });
}
function p(s, o = {}) {
  return new Paragraph({
    children: Array.isArray(s) ? s : [txt(s, o)],
    spacing: { before: o.before || 0, after: o.after === undefined ? 60 : o.after, line: 230 },
    alignment: o.align,
    indent: o.indent,
  });
}
function h1(s) {
  return new Paragraph({
    children: [txt(s, { bold: true, size: 21 })],
    spacing: { before: 200, after: 90 },
    border: { bottom: RULE },
  });
}
function cell(children, o = {}) {
  return new TableCell({
    width: { size: o.w, type: WidthType.DXA },
    shading: o.head ? { type: ShadingType.CLEAR, fill: GREY, color: 'auto' } : undefined,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    columnSpan: o.span,
    children: (Array.isArray(children) ? children : [children]).map((c) =>
      typeof c === 'string'
        ? new Paragraph({ children: [txt(c, { bold: o.head, size: o.size || 16 })], spacing: { after: 0, line: 220 } })
        : c),
  });
}
// rows: array of arrays of strings; widths: column widths summing to W
function table(widths, rows, opts = {}) {
  return new Table({
    columnWidths: widths,
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    rows: rows.map((r, i) =>
      new TableRow({
        children: r.map((c, j) => cell(c, { w: widths[j], head: opts.header && i === 0, size: opts.size })),
        tableHeader: opts.header && i === 0,
      })),
  });
}

const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: 17 } } } },
  sections: [{
    properties: { page: { margin: { top: 850, bottom: 700, left: 850, right: 850 } } },
    children: [
      new Paragraph({
        children: [txt('PARC 2026 予選 提出レポート（Track 1）', { bold: true, size: 26 })],
        spacing: { after: 40 }, alignment: AlignmentType.CENTER,
      }),
      new Paragraph({
        children: [txt('OpenVLA-OFT+ を追加学習なしで用い、推論時の平均化のみでスコアを構成した提出', { size: 16, color: '555555' })],
        spacing: { after: 140 }, alignment: AlignmentType.CENTER,
      }),

      // ---------------------------------------------------------------- 1
      h1('1. 学習・推論時の工夫点および試行錯誤'),

      p('当初は SmolVLA を用いたが、8 エピソード中 6 本がゴールに到達しない失敗モードに手が届かなかった。LIBERO-plus の報告値ではカメラ視点の摂動が全モデルの急所であり、そこだけ OpenVLA-OFT+ が突出していた（Camera 92.8 / Total 79.6）ため、ベースモデルを OpenVLA-OFT+ へ入れ替えた。以降は追加学習を行わず、推論側のみで改善した。'),

      p([txt('得られた改善はすべて「予測を平均する」方向から来た。', { bold: true }), txt('しかも独立な 2 方向がある。')]),

      table([1560, 5400, 900, 1980], [
        ['工夫', '内容', '寄与', '備考'],
        ['ベース変更', 'SmolVLA → OpenVLA-OFT+（LIBERO-plus mix-SFT 済み）', '+0.105', '0.188 → 0.293 相当'],
        [['時間方向の平均', '(temporal ensembling)'], 'ACT 方式。毎ステップ推論し、過去 8 回ぶんの「現ステップに対する予測」を重み w_i=exp(−m·i)（m=0.01）で加重平均。gripper 次元のみ平均せず最新値を使用', '+0.111', '単独で最大。0.144 → 0.293'],
        [['空間方向の平均', '(crop scale TTA)'], '中心クロップ倍率を変えた 5 視点（0.90 / 0.95 / 0.85 / 0.925 / 0.875）で推論し平均。平均は gripper 変換の前に取る（変換後は ±1 に二値化済みで平均が壊れるため）', '+0.078', '0.293 → 0.371'],
      ], { header: true, size: 15 }),

      p([txt('※ 寄与は各要素を単独で外した際の差分（他要素は最終構成に固定）であり、測定の基準が異なるため加算はできない。', { size: 15, color: '555555' })], { before: 60, after: 60 }),

      p([txt('参照構成から離れる変更は例外なく負けた。', { bold: true }), txt('クロップ中心を 0.900 → 0.875 に動かすと −0.012、ensembling の窓を 8 → 4 に狭めると −0.204、TTA に恒等クロップ 1.00 を混ぜると −0.033。倍率列の平均が学習時のクロップ（0.9）から離れることが損失の実体であり、0.900 は勾配上の点ではなく極値だった。')], { before: 40 }),

      p([txt('ヒューリスティックはすべて 0.000 だった。', { bold: true }), txt('把持失敗後の開き直し、停滞検知による視点切替、エピソード冒頭の待機の 3 つを実装し提出したが、いずれもスコアを 1 桁も動かさなかった。発火はしていたが、残る失敗の種類（そもそも把持を試みない／物体を掴んだまま完了しない）がヒューリスティックの射程外だった。冒頭の待機は評価ハーネス側が既に 10 ステップ実施しており、こちらの実装は二重適用だった。')]),

      p([txt('測定の方法論。', { bold: true }), txt('採点は決定的である（同一 zip の再提出で step 数まで完全一致）一方、ローカル評価はエピソード単位で再現しない（同一設定間で 39 本中 3 本が相違）。したがってローカルは退行検知にのみ用い、採否は必ず採点結果で判断した。採点ログの POST /reset と POST /act を数えるとエピソードごとの step 数が復元でき（300 = 時間切れ = 失敗）、合計スコアしか返らない採点から 8 本ぶんの内訳を得る手段として全期間で使用した。')]),

      p('最終成績は 16 回の提出で 0.000 → 0.371、8 エピソード中 4 本成功。残り 4 本はモデルの能力そのものに起因し、推論側の調整では到達できなかった。'),

      // ---------------------------------------------------------------- 2
      h1('2. モデル情報'),

      table([2200, 7640], [
        ['モデル名・バージョン', '提出システム: OpenVLA-OFT+ + temporal ensembling (h=8) + crop-scale TTA (5 視点)。構成識別子 tta5（parc_env: PARC_ENSEMBLE=1 / PARC_ENS_H=8 / PARC_ENS_GRIPPER=0 / PARC_OFT_TTA=5）'],
        ['ベースモデル名', 'OpenVLA-OFT+ … Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata（さらにその基盤は openvla/openvla-7b）'],
        ['ベース重みの入手元', 'Hugging Face Hub。huggingface_hub.snapshot_download で取得（tools/fetch_model.sh）。取得日 2026-08-11'],
        ['重みの revision / commit hash', '【要記入】revision 未指定で取得したため main ブランチの当時の HEAD。正確な commit hash は取得先の .cache/huggingface/download メタデータより取得のこと'],
        ['提出チェックポイントのハッシュ値', '提出 zip（submission_oft_tta5.zip）の SHA-256: a2cc2f91ab7b77a423f65f306ef8d1fceccdd953df6c2c71128bff84088a37ea'],
        ['モデル構成', 'Prismatic VLM 系 7B（dinosiglip 視覚バックボーン＋7B LLM、hidden 4096）、bf16、約 15.1 GB。行動ヘッドは L1 回帰ヘッド（自己回帰トークン生成ではなく決定的）。proprio projector により 8 次元の固有受容感覚を入力に併合'],
        ['推論方式', '1 回の forward で action chunk を一括生成する決定的推論。画像は両カメラとも 180 度回転 → 中心 90% クロップ → LANCZOS で 224×224。出力 gripper 次元は [0,1]→[−1,1] 正規化・sign による二値化・符号反転（環境規約）を適用。サンプリング・温度パラメータは無し'],
        ['Action chunk', 'あり。chunk 長 8（NUM_ACTIONS_CHUNK = 8）、行動次元 7。ただし chunk をそのまま開ループ実行はせず、毎ステップ再推論して上記 temporal ensembling で 1 ステップぶんに合成する'],
      ], { size: 15 }),

      // ---------------------------------------------------------------- 3
      h1('3. 学習情報'),

      p([txt('提出モデルに対する追加学習は行っていない。', { bold: true }), txt('公開チェックポイントをそのまま用い、改善はすべて推論時の処理による。以下は試行したが不採用となった学習の記録である。')]),

      table([2200, 7640], [
        ['使用した学習データ', 'lerobot/libero_plus（revision f3f49f426d75030177b18778374005bc12ccd588）。独自収集・生成データは無し'],
        ['データ件数・タスク数', '10 タスク × 60 エピソード / タスク を抽出して使用'],
        ['学習方法・区分', 'LoRA（PEFT）。対象は SmolVLA（lerobot/smolvla_libero_plus）であり、最終提出の OpenVLA-OFT+ ではない'],
        ['学習ステップ数', '15,000 steps'],
        ['学習対象パラメータ', 'LoRA アダプタのみ（rank 8）。ベース重みは凍結'],
        ['主なハイパーパラメータ', 'r=8 / batch size 32 / lr 1e-4 → 1e-5（decay）/ warmup あり'],
        ['結果と不採用の理由', '成功率が改善しなかったため棄却。原因は特定済みで、損失の 94% が行動のゼロ埋め次元（max_action_dim=32 に対し実次元 7）に向かい、rank 8 の容量がそこで消費されていた。当該次元を勾配から外すと破壊（−37.5〜−45 pt）は止まったが、実次元の損失が半減（0.2522 → 0.1228）しても成功率は 1 pt も動かなかった。ベースが既に解けるデータへの当てはめを改善しても、解けないタスクへの汎化は得られないと判断した'],
      ], { size: 15 }),

      // ---------------------------------------------------------------- 4
      h1('4. 権利関係'),

      table([2200, 7640], [
        ['ベースモデルのライセンス', '【要確認】Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata のモデルカード記載を提出時に確認のこと。基盤である OpenVLA および OpenVLA-OFT（moojink/openvla-oft）はいずれも MIT License'],
        ['学習データのライセンス', 'LIBERO: MIT License（Copyright (c) 2023 Lifelong Robot Learning）。LIBERO-plus（sylvestf/LIBERO-plus, および HF datasets/Sylvest/LIBERO-plus）: 本レポート作成時点で LICENSE ファイル・README ともにライセンス記載を確認できず。明示的許諾の無い著作物は既定で全権利が留保される点に留意。ただし LIBERO 由来部分には LIBERO の MIT License が及ぶ'],
        ['第三者コードのライセンス', '提出物に同梱: transformers 4.40.1 / tokenizers 0.19.1 / timm 0.9.10 / accelerate（いずれも Apache-2.0）、PyTorch（BSD-3-Clause）、NumPy（BSD-3-Clause）、FastAPI（MIT）、uvicorn（BSD-3-Clause）。submission/vendor_oft/ は checkpoint 同梱の trust_remote_code コードが必要とする prismatic.vla.constants / prismatic.training.train_utils の最小移植で、値は openvla-oft（MIT License）の LIBERO 設定に一致させている。評価環境側で使用: LIBERO（MIT）、robosuite 1.4.0（MIT）、MuJoCo 3.7.0（Apache-2.0）、gym 0.25.2（MIT）ほか。全一覧は同梱の THIRD_PARTY_LICENSES.md に記載'],
      ], { size: 15 }),
    ],
  }],
});

Packer.toBuffer(doc).then((b) => {
  fs.writeFileSync('/home/user/PARC2026_pre/report/PARC2026_report.docx', b);
  console.log('written');
});
