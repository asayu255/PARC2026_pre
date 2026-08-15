const {
  Document, Packer, Paragraph, TextRun, AlignmentType, LineRuleType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
} = require('docx');
const fs = require('fs');
const path = require('path');

const FONT = 'Yu Gothic';
const W = 10306;           // 本文幅 (DXA)。左右余白 800 ずつの A4
const GREY = 'F2F2F2';
const RULE = { style: BorderStyle.SINGLE, size: 4, color: 'AAAAAA' };

// 行送りは必ず EXACT で指定する。Yu Gothic は hhea の lineGap が大きく、
// auto のままだと 1 行が 1.5em 近くになってページ数が読めない。ここを
// 固定しておくと estimate_pages.py の見積りが実寸と一致する。
const LINE_BODY = 215;     // 8.5pt 本文
const LINE_TBL = 188;      // 7.5pt 表
const LINE_HEAD = 260;     // 10.5pt 見出し

function txt(s, o = {}) {
  return new TextRun({ text: s, font: FONT, size: o.size || 17, bold: !!o.bold, color: o.color });
}
function p(s, o = {}) {
  return new Paragraph({
    children: Array.isArray(s) ? s : [txt(s, o)],
    spacing: {
      before: o.before || 0,
      after: o.after === undefined ? 55 : o.after,
      line: o.line || LINE_BODY,
      lineRule: LineRuleType.EXACT,
    },
    alignment: o.align,
  });
}
function h1(s) {
  return new Paragraph({
    children: [txt(s, { bold: true, size: 21 })],
    spacing: { before: 150, after: 70, line: LINE_HEAD, lineRule: LineRuleType.EXACT },
    border: { bottom: RULE },
  });
}
function h2(s) {
  return new Paragraph({
    children: [txt(s, { bold: true, size: 17 })],
    spacing: { before: 110, after: 45, line: LINE_BODY, lineRule: LineRuleType.EXACT },
  });
}
function cell(children, o = {}) {
  return new TableCell({
    width: { size: o.w, type: WidthType.DXA },
    shading: o.head ? { type: ShadingType.CLEAR, fill: GREY, color: 'auto' } : undefined,
    margins: { top: 35, bottom: 35, left: 80, right: 80 },
    children: (Array.isArray(children) ? children : [children]).map((c) =>
      new Paragraph({
        children: [txt(c, { bold: o.head, size: o.size || 15 })],
        spacing: { after: 0, line: LINE_TBL, lineRule: LineRuleType.EXACT },
      })),
  });
}
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
const note = (s) => p([txt(s, { size: 14, color: '555555' })], { before: 40, after: 55, line: 180 });

const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: 17 } } } },
  sections: [{
    properties: { page: { margin: { top: 760, bottom: 620, left: 800, right: 800 } } },
    children: [
      new Paragraph({
        children: [txt('PARC 2026 予選 提出レポート（Track 1）', { bold: true, size: 26 })],
        spacing: { after: 30, line: 320, lineRule: LineRuleType.EXACT }, alignment: AlignmentType.CENTER,
      }),
      new Paragraph({
        children: [txt('OpenVLA-OFT+ を追加学習なしで用い、推論時の平均化のみでスコアを構成した提出（最終 0.371）', { size: 15, color: '555555' })],
        spacing: { after: 110, line: 200, lineRule: LineRuleType.EXACT }, alignment: AlignmentType.CENTER,
      }),

      // ================================================================ 1
      h1('1. 学習・推論時の工夫点および試行錯誤'),

      h2('1.1 ベースモデルの選定'),
      p('当初は SmolVLA を用いたが、8 エピソード中 6 本がゴールに到達しない失敗モードに手が届かなかった。LIBERO-plus の報告値ではカメラ視点の摂動が全モデルの急所であり（OpenVLA 0.8 / π₀ 13.8 / π₀-Fast 65.1）、そこだけ OpenVLA-OFT+ が 92.8 と突出していた。PARC が使う摂動分布そのもので mix-SFT されている点が決め手で、ベースモデルを入れ替えた。以降は追加学習を行わず、推論側のみで改善している。'),

      h2('1.2 決定的な行動ヘッドが「平均」の設計を規定した'),
      p('OpenVLA-OFT+ の行動ヘッドは L1 回帰であり、自己回帰的なトークン生成ではない。つまり同じ観測からは常に同じ chunk が出る。したがって拡散モデルのように「同じ入力で複数回サンプリングして平均する」ことができず、平均を作るには入力側を変えるしかない。ここから独立な 2 方向が出てきて、結果的にスコアの伸びはほぼすべてこの 2 つから来た。'),

      table([1500, 5600, 900, 2306], [
        ['工夫', '内容', '寄与', '備考'],
        ['ベース変更', 'SmolVLA → OpenVLA-OFT+（LIBERO-plus mix-SFT 済み）', '+0.105', '0.188 → 0.293'],
        ['時間方向の平均', 'ACT の temporal ensembling。chunk を開ループ実行せず毎ステップ推論し、過去 8 回ぶんの「現ステップに対する予測」を w_i=exp(−m·i)（m=0.01）で加重平均。gripper 次元のみ平均せず最新値を採用', '+0.111', '単独で最大。0.144 → 0.293'],
        ['空間方向の平均', '中心クロップ倍率を変えた 5 視点（0.90 / 0.95 / 0.85 / 0.925 / 0.875）で推論し平均。平均は gripper 変換の前に取る（変換後は ±1 に二値化済みで、平均すると中間値になり二値化が壊れる）', '+0.078', '0.293 → 0.371'],
      ], { header: true }),
      note('※ 寄与は各要素を最終構成から外した際の差分であり、測定の基準が異なるため加算はできない。'),

      h2('1.3 TTA は視点数ではなく「倍率列の平均位置」で決まっていた'),
      p('視点を増やすほど良くなると考えて 2 → 4 視点に増やしたところ、0.304 → 0.271 と下がった。3 視点に組み直すと 0.370 に跳ねた。視点数と単調でないので、当初の理解は誤りだった。'),
      p('エピソード単位で分解すると原因が見えた。2 視点（0.90, 0.95 ＝ 平均 0.925）では ep6 が 119 → 214 step に伸びており、3 視点（0.90, 0.95, 0.85 ＝ 平均 0.900）で 119 step に戻っている。ep1 の −61 step と合わせて、この 2 本で改善分のほぼ全部を説明できる。4 視点が負けたのは恒等クロップ 1.00 を混ぜていたためで、これは学習時のクロップ（0.9）から最も遠い。'),
      p([txt('つまり効いていたのは視点数ではなく、倍率列の平均が学習時の 0.9 に一致していることだった。', { bold: true }), txt('以後は倍率列を 0.90 を中心に対称に組み、先頭から何視点を取っても平均が 0.9 に保たれるよう設計した。これを確かめるため中心自体を 0.875 へ動かす提出も行ったが −0.012 で、0.900 が勾配上の点ではなく極値であることが確定した。')]),

      h2('1.4 効かなかった軸'),
      p('参照構成から離れる変更は例外なく負けた。以下はすべて実測して棄却したものである。'),
      table([2400, 1100, 6806], [
        ['試した軸', '結果', '内容・棄却の理由'],
        ['ensembling の窓 h', '−0.204', '8 → 4 に狭めると成功が 4 本 → 2 本に減少。平均する本数がそのまま効いている'],
        ['クロップ中心', '−0.012', '0.900 → 0.875。学習時のクロップからの変位が損失の実体'],
        ['恒等クロップの混入', '−0.033', 'TTA の倍率列に 1.00 を入れる。最も学習時から遠い視点で毒になる'],
        ['振幅スケール / slew 制限', '棄却', '並進のみ・回転のみの振幅スケールと、1 step の変化量上限。いずれも局所評価で改善せず、提出枠を使う価値なしと判断'],
        ['ヒューリスティック 3 種', '±0.000', '把持失敗後の開き直し（ep4 で 2 回発火）、停滞検知による視点切替（ep3 で 2 回・ep4 で 1 回）、冒頭の待機。いずれも発火したがスコアは動かず。待機は評価ハーネス側が既に 10 step 実施済みで二重適用だった'],
        ['LoRA 追加学習 / 別系統', '棄却', 'LoRA は 3 回連続で成功率が −37.5〜−45 pt（原因は §3）。π₀ / π₀.5 と SmolVLA 継続は LIBERO-plus の報告値で OFT+ に劣後'],
      ], { header: true }),

      h2('1.5 残った失敗は 4 本すべてヒューリスティックの射程外だった'),
      p('最終提出（tta5）の 8 本を採点ログから分解すると、成功 4 本（147 / 110 / 110 / 119 step）に対し、失敗 4 本の内訳は次のとおりで、種類が互いに異なる。'),
      table([700, 2650, 6956], [
        ['本', '観測', '解釈'],
        ['ep3', '300 step 中、閉指令が 0 回', 'そもそも把持を試みていない。把持の後処理をいくら足しても届かない'],
        ['ep4', '閉指令 224 回、指の開き 0.0010', '完全に閉じて中身が無い＝空掴み。開き直しを 2 回発火させても同じ場所で再現した'],
        ['ep7', '閉指令 267 回、開き 0.0044〜0.0066', '物体を保持したまま 300 step 完了しない。把持は成立しており、失敗はその先にある'],
        ['ep8', '全 16 回の提出で一度も成功せず', '唯一 SmolVLA 版が成功させたことがあり、モデルの得手不得手の差と考えられる'],
      ], { header: true }),
      p('ep3 は把持を試みないので把持のヒューリスティックが起動せず、ep4 は開き直しても同じ空掴みを繰り返し、ep7 に至っては把持自体が成立している。残る 4 本はモデルの能力そのものに起因し、推論側の後処理では到達できないというのが結論である。'),

      h2('1.6 採点式を逆解析して提出枠の配分を決めた'),
      p('採点から返るのは合計スコア 1 個だけだが、ログの POST /reset と POST /act を数えるとエピソードごとの step 数が復元できる（300 = 時間切れ = 失敗）。これを全 12 回のログに適用して、採点式を次の形に当てた。'),
      p([txt('　score = Σ(成功したエピソードの smooth) / 8　、　smooth ≈ 1.129 − 0.003209 × step', { size: 16 })], { after: 45 }),
      p('ensembling 世代 5 回に対し 2 パラメータで最大残差はスコア換算 0.005。これにより「成功を 1 本増やす = +0.067〜0.093」「成功 4 本を各 10 step 短縮 = +0.016」と事前に見積もれるようになり、つまみを 1 段ずつ提出して +0.005 を確かめる使い方をやめ、失敗の種類が変わりうる変更にだけ提出枠を充てる方針に切り替えられた。'),
      p([txt('ただしこの直線は細かい差では符号を誤る。', { bold: true }), txt('最終盤に 4 桁目まで見えたことで判明したもので、tta5 は tta3 より成功本の合計 step が 2 step 長いにもかかわらず 0.3710 対 0.3699 で勝っていた。直線は tta3 が 0.0008 勝つと予測する。合成スコアには step 数と独立な滑らかさの項（jerk・sparc・path length）があり、視点を増やして軌道が滑らかになった分がそこに乗ったと解釈している。大きな差では残差 0.005 以内で当たっており判断には使えていたが、粗い道具であって精密な道具ではない。')]),

      h2('1.7 測定の作法と提出運用'),
      p('採点は決定的である（同一 zip を再提出したところ step 数まで完全に一致した）一方、ローカル評価はエピソード単位で再現しない（同一設定間で 39 本中 3 本が相違）。したがってローカルは退行検知にのみ用い、採否は必ず採点結果で判断した。'),
      p('運用面では、構成違いが parc_env 1 行しか違わないのに 12 GB を書き直すのが無駄だったため、zip 内の 1 エントリだけを差し替える手順を用意した。提出前には HF キャッシュを空にした状態で起動・レイテンシ・決定性を検証している。'),

      // ================================================================ 2
      h1('2. モデル情報'),
      table([2150, 8156], [
        ['モデル名・バージョン', '提出システム: OpenVLA-OFT+ + temporal ensembling (h=8) + crop-scale TTA (5 視点)。構成識別子 tta5（parc_env: PARC_ENSEMBLE=1 / PARC_ENS_H=8 / PARC_ENS_GRIPPER=0 / PARC_OFT_TTA=5）'],
        ['ベースモデル名', 'OpenVLA-OFT+ … Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata（その基盤は openvla/openvla-7b）'],
        ['ベース重みの入手元', 'Hugging Face Hub。huggingface_hub.snapshot_download で取得（tools/fetch_model.sh）。取得日 2026-08-11'],
        ['重みの revision / commit hash', 'commit a85655ec941bae6644c9fbdf62db02b9726d7cf5（main ブランチ、2026-08-11 取得時点の HEAD）。revision を指定せず取得したため、ローカルの download メタデータから復元した。全ファイルが同一 commit を指しており、スナップショットの整合は確認済み'],
        ['提出チェックポイントのハッシュ値', 'SHA-256: a2cc2f91ab7b77a423f65f306ef8d1fceccdd953df6c2c71128bff84088a37ea（submission_oft_tta5.zip、約 12 GB）'],
        ['モデル構成', 'Prismatic VLM 系 7B（dinosiglip 視覚バックボーン＋7B LLM、hidden 4096）、bf16、約 15.1 GB。行動ヘッドは L1 回帰ヘッドで決定的（自己回帰トークン生成ではない）。proprio projector により 8 次元の固有受容感覚を入力に併合'],
        ['推論方式', '1 回の forward で action chunk を一括生成する決定的推論。画像は両カメラとも 180 度回転 → 中心 90% クロップ → LANCZOS で 224×224。gripper 次元は [0,1]→[−1,1] 正規化・sign による二値化・符号反転（環境規約）を適用。サンプリング・温度パラメータは無し。1 リクエストあたり約 1.34 秒、8 エピソードで合計 2,300 秒'],
        ['Action chunk', 'あり。chunk 長 8（NUM_ACTIONS_CHUNK = 8）、行動次元 7。ただし chunk を開ループ実行はせず、毎ステップ再推論して上記 temporal ensembling で 1 ステップぶんに合成する'],
      ]),

      // ================================================================ 3
      h1('3. 学習情報'),
      p([txt('提出モデルに対する追加学習は行っていない。', { bold: true }), txt('公開チェックポイントをそのまま用い、改善はすべて推論時の処理による。以下は試行したが不採用となった学習の記録である。')]),
      table([2150, 8156], [
        ['使用した学習データ', 'lerobot/libero_plus（revision f3f49f426d75030177b18778374005bc12ccd588）。独自に収集・生成したデータは無し'],
        ['データ件数・タスク数', '10 タスク × 60 エピソード / タスク を抽出して使用'],
        ['学習方法・区分', 'LoRA（PEFT）。対象は SmolVLA（lerobot/smolvla_libero_plus）であり、最終提出の OpenVLA-OFT+ ではない'],
        ['学習ステップ数', '15,000 steps'],
        ['学習対象パラメータ', 'LoRA アダプタのみ（rank 8）。ベース重みは凍結'],
        ['主なハイパーパラメータ', 'r=8 / batch size 32 / lr 1e-4 → 1e-5（decay）/ warmup あり'],
        ['結果と不採用の理由', '成功率が改善しなかったため棄却。原因は特定済みで、損失の 94% が行動のゼロ埋め次元（max_action_dim=32 に対し実次元 7）に向かい、rank 8 の容量がそこで消費されていた。当該次元を勾配から外すと破壊（−37.5〜−45 pt）は止まったが、実次元の損失が半減（0.2522 → 0.1228）しても成功率は 1 pt も動かなかった。ベースが既に解けるデータへの当てはめを改善しても、解けないタスクへの汎化は得られないと判断した'],
      ]),

      // ================================================================ 4
      h1('4. 権利関係'),
      table([2150, 8156], [
        ['ベースモデルのライセンス', 'MIT License。Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata のモデルカードに全文が掲載されている（ただし著作権表示は "Copyright (c) [year] [fullname]" とテンプレートのままで、権利者名は未記入）。基盤である OpenVLA および OpenVLA-OFT（moojink/openvla-oft）も MIT License'],
        ['学習データのライセンス', 'LIBERO: MIT License（Copyright (c) 2023 Lifelong Robot Learning）。LIBERO-plus（sylvestf/LIBERO-plus および HF datasets/Sylvest/LIBERO-plus）: 本レポート作成時点で LICENSE ファイル・README ともにライセンス記載を確認できず。明示的許諾の無い著作物は既定で全権利が留保される点に留意。ただし LIBERO 由来部分には LIBERO の MIT License が及ぶ'],
        ['第三者コードのライセンス', '提出物に同梱: transformers 4.40.1 / tokenizers 0.19.1 / timm 0.9.10 / accelerate（いずれも Apache-2.0）、PyTorch（BSD-3-Clause）、NumPy（BSD-3-Clause）、FastAPI（MIT）、uvicorn（BSD-3-Clause）。submission/vendor_oft/ は checkpoint 同梱の trust_remote_code コードが要求する prismatic.vla.constants / prismatic.training.train_utils の最小移植で、値は openvla-oft（MIT License）の LIBERO 設定に一致させている。評価環境側で使用: LIBERO（MIT）、robosuite 1.4.0（MIT）、MuJoCo 3.7.0（Apache-2.0）、gym 0.25.2（MIT）ほか。全一覧は同梱の THIRD_PARTY_LICENSES.md に記載'],
      ]),
    ],
  }],
});

Packer.toBuffer(doc).then((b) => {
  fs.writeFileSync(path.join(__dirname, 'PARC2026_report.docx'), b);
  console.log('written');
});
