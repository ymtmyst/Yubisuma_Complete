# Complete AI チェックリスト・進捗管理票

更新日: 2026-05-20  
対象: `complete_ai_plan.html` に基づく Complete ルール最強 AI 作成計画  
判定基準: リポジトリ内の実装、テスト、生成済みレポートを確認して評価

## 運用ルール

今後 Complete ルール AI 関連の作業を行った場合は、作業完了時にこのファイルも更新すること。

- 実装、テスト、レポート生成、仕様整理、調査のいずれかを進めたら、該当する `進捗管理票` の進捗率・状態・完了済み・残タスクを更新する。
- チェックリスト項目を完了した場合は、対応する `- [ ]` を `- [x]` に変更する。
- 新しい作業項目が発生した場合は、該当フェーズの残タスクまたはチェックリストへ追加する。
- テストやレポート生成を実行した場合は、`確認済みコマンド` にコマンドと結果を追記する。
- 進捗率を変更する場合は、実装済みファイル、生成済み成果物、テスト結果のいずれかに基づいて判断する。
- 大きな方針変更やルール解釈の決定があった場合は、`次にやる順番` と関連フェーズの完了条件も見直す。

## 現状サマリー

完成度の見立ては次の通り。

- 最初のマイルストーン達成度: 75%
- Complete-lite exact solver 周辺: 55%
- Full Complete 学習 AI まで含めた全体完成度: 28%

現在は「計画書だけ」の段階ではなく、`complete_solver/` に純粋な状態表現、合法手生成、1ターン遷移、深さ制限 subgame solver、CSV/HTML レポート出力が入っている。`results/complete_lite/` と `results/complete_lite_depth2/` には代表局面の depth 1/2 結果も生成済み。

ただし、まだ「最強 AI」本体ではない。現状は、学習 AI の土台になる Complete-lite の厳密/準厳密解析基盤で、Gymnasium 環境、PPO/NFSP/Deep CFR/R-NaD、探索統合、4構成リーグ評価は未着手。

## 進捗管理票

| ID | フェーズ | 進捗 | 状態 | 完了済み | 残タスク | 完了条件 |
|---|---:|---:|---|---|---|---|
| P0 | 調査・計画書 | 100% | 完了 | `complete_ai_plan.html` 作成済み。優先課題、ロードマップ、評価指標、実装ファイル案が整理済み。 | 進捗に応じた更新。 | 計画と実装方針が確認可能。 |
| P1 | 仕様固定と solver-facing spec | 65% | 進行中 | `complete_solver/rules_spec.md` 作成済み。`docs/rules_decisions.md` で視点、同時行動、合法手、参照/ストック、リバーシ交換対象、Mirror 反射対象を固定。参照用 `docs/rules_interpretation.html` を追加。 | 各スキルの曖昧点を golden test 化。開幕制限、タイム、ミラーの詳細明文化。 | ルール文書から独立して solver 実装を検証できる仕様とテストが揃う。 |
| P2 | 副作用なし Pure Rules Engine | 78% | 進行中 | `state.py`、`actions.py`、`transition.py` で dataclass ベースの状態、合法手、1ターン遷移を実装。print/input/乱数なし。Choice/All/Drop、必殺、Mirror/Reversi の golden test を追加。Mirror は数字/フラッシュ/セメント/ドロップだけを反射し、参照系は参照元準拠に修正。 | 既存対話実装との網羅照合。複合効果、開幕制限の追加テスト。 | 同じ `state, joint_action` から常に同じ `Transition` が返り、代表ルールがテストで固定される。 |
| P3 | 合法手生成・有限ストック | 75% | 進行中 | TP/NTP の合法手生成あり。ストックは `frozenset` で重複不可。Mirror OFF/ON のストック可能数差分に対応する基礎あり。既存 `get_valid_skills` との代表状態照合テストを追加。 | Choice/All の後出し性や順序表現の妥当性検証。NTP reaction 側の照合/仕様固定。 | 状態依存の行動列挙が既存仕様と一致し、ストック上限が常に守られる。 |
| P4 | Mirror/Reversi 対応 | 62% | 進行中 | `RulesConfig(enable_mirror, enable_reversi)` と代表的なミラー/リバーシ遷移テストあり。Mirror の可/不可分類、対カウンタースキル不可、必殺不可、参照元準拠、All の複数反射、Reversi の交換対象/非対象をテスト化。 | 4構成共通テスト、既存対話実装側への反映。 | Mirror/Reversi の ON/OFF 4構成で同じテストスイートが通る。 |
| P5 | Complete-lite exact solver | 55% | 進行中 | `matrix_game.py` の LP、`finite_horizon.py` の深さ制限 subgame solver、深さ1/2の代表局面レポートが動作。 | 状態全列挙、価値反復、`gamma=0.995〜0.9995` 感度分析、収束ログ、支配行動除去。 | 初期局面と代表局面の均衡方策・価値が再現可能に出力される。 |
| P6 | レポート・可視化 | 60% | 進行中 | `reports.py`、CSV、`index_depth1.html`、`index_depth2.html` 作成済み。サニティCSVで主要スキル質量を確認可能。 | HTML/CLI の一部日本語文字化け修正。`.ini` 戦術解説との定性整合チェックを表に追加。 | 人間が局面価値、混合方策、定性コメントを読めるレポートになる。 |
| P7 | 終盤表・戦術表 | 25% | 未完了寄り | `locked_flash`、`endgame_number`、`charge_number`、`quick_followup` など代表シナリオはあり。 | 残り手1/2、セメント固定、ガード、チャージ、クイック、ロックなどの体系的な局面表を生成。 | 探索の葉評価と教師データに使える局面表が揃う。 |
| P8 | 全 Complete 環境 | 0% | 未着手 | なし。 | `complete_rl/env.py`、action mask、observation、自己対戦 API を作成。 | Gymnasium 互換で 4構成を切り替えられる。 |
| P9 | 学習 AI | 0% | 未着手 | `requirements.txt` に torch/sb3/gymnasium 等の候補あり。 | MaskablePPO、NFSP、Deep CFR、Expert Iteration、R-NaD の実装/比較。 | 勝率だけでなく exploitability/NashConv で改善を追える。 |
| P10 | 探索統合 | 0% | 未着手 | なし。 | 方策を事前分布にした局所 subgame 探索、詰み/必殺/ミラー/リバーシ局面の深掘り。 | 実戦時に重要局面で読みを深くできる。 |
| P11 | 4構成統合評価 | 10% | 未着手寄り | Config フラグと一部テストの土台はあり。 | Mirror/Reversi 4構成のリーグ戦、構成別モデル vs 単一モデルの比較。 | 4構成を同一基盤で評価し、採用方針を決められる。 |

## チェックリスト

### 最初のマイルストーン

- [x] Complete solver-facing spec の初版を作る。
- [x] 対話用実装と分離した `complete_solver/` を作る。
- [x] `State` / `PlayerState` を immutable dataclass として定義する。
- [x] TP/NTP の合法手生成を実装する。
- [x] 副作用なしの 1ターン遷移を実装する。
- [x] 数字、フラッシュ、カウンター、ストック、オール、ミラー、リバーシの代表テストを作る。
- [x] LP によるゼロサム行列ゲーム solver を実装する。
- [x] depth-limited subgame solver を実装する。
- [x] 代表局面の CSV/HTML レポートを生成する。
- [x] 仕様差分を `docs/rules_decisions.md` 形式で固定する。
- [x] 既存 `get_valid_skills` と合法手生成を照合する。
- [ ] 各スキルの golden test を網羅する。（Choice/All/Drop、必殺、Mirror/Reversi の代表ケースは追加済み）
- [ ] CLI/HTML レポート内の日本語文字化けを修正する。

### Complete-lite exact solver

- [x] Mirror OFF / Reversi OFF の基本解析が動く。
- [x] depth 1 の `initial`、`locked_flash`、`stock_choice`、`guarded`、`endgame_number`、`charge_number`、`quick_followup` を出力する。
- [x] depth 2 の `initial`、`locked_flash`、`endgame_number` を出力する。
- [ ] 状態全列挙を実装する。
- [ ] 割引価値反復を実装する。
- [ ] `gamma=0.995〜0.9995` の感度分析を出す。
- [ ] 状態数、収束回数、収束誤差をレポートに出す。
- [ ] 支配行動除去または行動枝刈りを入れる。
- [ ] `.ini` 戦術解説との整合チェックをレポート化する。

### Full Complete への拡張

- [ ] Mirror ON / Reversi OFF の regression suite を作る。
- [ ] Mirror OFF / Reversi ON の regression suite を作る。
- [ ] Mirror ON / Reversi ON の regression suite を作る。
- [ ] 4構成すべてで同一シナリオレポートを生成する。
- [ ] Gymnasium 互換環境を作る。
- [ ] 行動マスクを実装する。
- [ ] 観測設計を固定する。
- [ ] MaskablePPO の自己対戦 baseline を作る。
- [ ] NFSP / Deep CFR / Expert Iteration / R-NaD の候補を比較する。
- [ ] exploitability / NashConv / exact subgame KL を評価指標として実装する。
- [ ] 探索統合を行う。
- [ ] 4構成リーグ評価を行う。

## 確認済みコマンド

```powershell
python -m unittest discover complete_solver\tests
```

結果: 41 tests OK

`pytest` は現在の Python 環境に未導入だったため、標準の `unittest` で確認した。

## 次にやる順番

1. CLI/HTML レポート内の日本語文字化けを修正する。
2. Mirror reaction の残りスキルと、開幕制限/複合効果の golden test を追加する。
3. 4構成の batch report を生成できる CLI オプションまたはスクリプトを追加する。
4. 状態列挙と価値反復に進む。
