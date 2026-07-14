# 次アクション引き継ぎ: graph-BR 列挙バッチ化 → D7–8 到達

作成 2026-07-14(セッション復旧時)。この1タスクだけを確実に引き継ぐためのメモ。
関連: `WORK_LOG.md` の graph-BR 各節 / メモリ `project-complete-ai`。

---

## ゴール(なぜやるか)
- **目的**: graph-BR(厳密 best-response による exploitability)を **D=7–8** まで押し上げ、
  「本採用 graph-vi モデルがどこまで搾取されうるか」の**深部**を確定する。
- **ユーザー最優先軸**: exploitability = 神視点最適解への近さ。head-to-head ではなくこれが成長の主指標。
- **未解決の核心**: 「ストック/序盤の長期スキルでモデルが搾取されるのでは?」というユーザー当初懸念の
  **唯一の決着手段**。現状 D4 0.5169 / D5 0.5283 / D6 0.5368 と**単調上昇・頭打ちなし**(外挿真値 ~0.55–0.56)。
  深さで搾取が増える=長期筋の搾取存在を示唆。D7–8 で「頭打ちの確認」と「ストック起因の切り分け」がしたい。
  ※ D値は **fast(定義版)エンジン**の数値(下記)。旧 enumerate_br の D4=0.5169 とは基準が違う点に注意。

## 現状(ここまで出来ている)
- `complete_ai/graph_br_fast.py` の **`enumerate_solve_fast`** = レベル同期BFS列挙。
  - 各レベルの凍結方策 solve を **per-thread サーチャ × `solve_batch` でスレッド並列**(`_parallel_solve`)。
  - 子生成は njit **`_grid_children`**、格納は畳み込みフラット配列(`_Grow` 成長バッファ、GB級対応)。
  - 深さ/終盤/終端/超過は**葉定数に畳み込み**、VI は内部エッジ(ci≥0)のみ走査。
- VI = **`solve_flat`**(njit 連立 Jacobi `_coupled_sweep`)。**250k状態で約55分→njitで数秒**。
- **正しさは担保済み**: njit VI == 純Python VI(`tests/test_graph_br_fast.py` トイ+fuzz)、
  列挙は **fast(D=2)==ref(D=2)** で担保(`test_fast_matches_reference_at_base_depth`)。
- 実測到達: D4 46s / D5 クリーン / D6 346,378状態 クリーン(hit_cap=False、VI njit 19s)。

## 【最重要】fast 列挙は「正典」— バグと誤解するな
- `enumerate_solve_fast`(BFS距離キャップ)と参照 `enumerate_br`(経路長キャップ)は
  **enum結果が意図的に一致しない**。fast=各局面を最短距離で1回展開し常にBR値を使う=**より正確なタイト下界**。
- **fast==ref は D=2 のみ**。D≥3 で fast が単調に上。状態数 fast(D).n ≈ ref(D+1).n。
- enum-vs-enum の parity で不一致を見て「off-by-one」と誤診して fast を ref に合わせる**修正をしてはいけない**
  (精度が落ちる)。詳細は `graph_br_fast.py` モジュール docstring と WORK_LOG「セッション復旧」節。

## 残る律速(D7–8 を阻むもの)
1. **凍結方策 solve のスループット**が支配(torch ネット + njit 探索機構 expand/backup/LP)。
   プロファイル(8000状態): solve が約70%(ネットbatch=1 17% / njit探索機構 28% / pincerA0辞書 9%)、
   列挙セル走査 23%。`solve_batch`(ネットプール)は実測 1.16× どまり = **ネットより njit 探索機構が非バッチで支配**。
2. **状態数の爆発**: D6 で 34.6万、D7–8 は数百万〜。メモリ(フラット配列は対応済みだが cap/RAM 要確認)と
   列挙時間(158–500 状態/s では数百万は非現実)。

## 具体的な打ち手(優先順・提案)
1. **njit 探索機構のバッチ化が本丸**(プロファイル的にネットバッチより効く)。凍結方策 solve は
   BatchedSearcher/PincerSearcher の depth-2 展開。**多状態の expand/backup/LP を局面またぎでバッチ njit 化**
   すれば solve のスループットが跳ねる。`batched_search.py` の `solve_batch` / `parallel_depth3_values` 系の
   nogil njit 資産を流用できるはず。
2. **列挙のスループット計測を先に取る**(`enumerate_solve_fast` の states/s を D5–D6 で実測 → ボトルネックが
   本当に solve か格納かを再確認してから着手)。
3. **cap / メモリの上限設計**: D7–8 の想定状態数を D4→D5→D6 の増加率から外挿し、cap と RAM を先に見積もる。
   閉じない(hit_cap=True)なら数値は過小評価になるので**必ず hit_cap=False を確認**して報告。
4. **収束の健全性**: `solve_flat` は omega ダンピング付き。D7–8 でも converged=True・max_delta<eps を確認。
5. (任意)**ストック切り分け**: 搾取がストック由来か見るなら prune_stock や状態フィルタで
   ストック無し部分集合の exploitability と比較。

## 実行/検証コマンド
- 単発: `python -m complete_ai.n7_graph_br --model models/value_gvi_latest.pt --max-depth 6 --engine njit`
  （注: CLI は `enumerate_br`+njit VI。**fast 列挙は `enumerate_solve_fast` を使う掃引スクリプト側**。
   D7–8 用の掃引ドライバは新規に書く/前回のインライン版を再構成する。verbose は絞ること=氾濫防止)
- テスト: `python -m unittest complete_ai.tests.test_graph_br_fast`(VI parity + D=2一致 + 単調性)
- 参考プロファイル: `data/graphbr_profile.txt`、D6ログ `data/graphbr_d6_njit.log`

## 落とし穴(必読)
- **Numba キャッシュ**: `graph_br_fast.py` の njit(`_coupled_sweep`/`_grid_children` 等)を変更したら
  `complete_ai/__pycache__` の該当 `.nbi/.nbc` を**必ず削除**(caller が旧 callee を焼き込む)。
- **verbose 氾濫**: 大規模ランは進捗出力を絞る/ログにリダイレクト。前回のセッション中断はこの氾濫が発端。
- **モデルに触れない**: graph-BR は測定専用。`value_gvi_latest.pt`(本採用)等は書き換え厳禁。
- **作業スタイル**: 着手時に「専門+平易+所要時間見積もり」、完了時に WORK_LOG 追記(メモリ `feedback-working-style`)。
- **本採用は graph-vi 継続**、最終決着はこの graph-BR finalize に委ねる(メモリ参照)。

## 完了の定義(DoD)
- D7 と(可能なら)D8 が **hit_cap=False・converged=True** で算出できる。
- exploitability の深さトレンド(D4..D8)が更新され、**頭打ちの有無**が言える。
- できれば**ストック起因の搾取**を切り分けた所見。
- WORK_LOG + 方策/exploitability レポート(レポートハブ登録)更新、テスト維持。
