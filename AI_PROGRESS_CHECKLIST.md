# Complete AI チェックリスト・進捗管理票

更新日: 2026-05-20（BC事前学習実装）  
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

- 最初のマイルストーン達成度: **99%**（支配行動除去のみ暫定後回し）
- Complete-lite exact solver 周辺: **85%**（状態全列挙・価値反復・感度分析まで完了）
- Full Complete 学習 AI まで含めた全体完成度: **64%**（P9 BC事前学習実装・113テスト PASS）

`complete_solver/` に純粋な状態表現・合法手生成・1ターン遷移・深さ制限 subgame solver・状態全列挙・割引価値反復・CSV/HTML レポートが揃っている。`complete_rl/` には Gymnasium 互換環境、action mask、MaskablePPO baseline CLI、smoke/quick/standard preset、複数 seed 評価、4構成 batch 学習/評価 CLI、名前付きNTP反応方策（none/counter_first/block_first/mirror_first/mixed_basic/weighted_none_counter）、保存済みモデルディレクトリの評価表生成（`--eval-dir`）、任意の報酬 shaping（`--reward-mode terminal|material`）、**BC事前学習（`bc_pretrain.py`、`--bc-pretrain` CLI フラグ）**がある。`results/complete_lite_v2/` に 12 シナリオ × depth 1 の結果を生成済み。gamma 感度分析（0.990〜0.9995）も完了し V(init)≈0.072 と安定。`quick` preset は seed 0/1 で実走し、random NTP 評価では 100 episode 追試も 100勝/0敗。4構成 quick seed 0 も実走し、NTP方策別評価表を `results/maskable_ppo_all_configs_quick_s0_eval/` に生成済み。`material` shaping pilot は `none` 相手の打ち切り解消に効いたが、`counter_first` には過適応による全敗が出た。`mixed_basic` / `weighted_none_counter` pilot は random/block に強いが none/counter の弱点が残った。テストは 113 件全 PASS。

ただし、まだ「最強 AI」本体ではない。MaskablePPO は quick preset までで、評価相手は random NTP に限られる。standard 長時間学習・NFSP/Deep CFR/R-NaD・探索統合・4構成リーグ評価は未着手。

## 進捗管理票

| ID | フェーズ | 進捗 | 状態 | 完了済み | 残タスク | 完了条件 |
|---|---:|---:|---|---|---|---|
| P0 | 調査・計画書 | 100% | 完了 | `complete_ai_plan.html` 作成済み。優先課題、ロードマップ、評価指標、実装ファイル案が整理済み。 | 進捗に応じた更新。 | 計画と実装方針が確認可能。 |
| P1 | 仕様固定と solver-facing spec | 80% | 進行中 | `docs/rules_decisions.md` で主要ルール固定済み。`test_compound_effects.py` で開幕制限・Guard・Charge×2・Copy+数字・Lock の golden test 追加。 | タイムの詳細挙動、ミラー×複合スキルの残り曖昧ケース明文化。 | ルール文書から独立して solver 実装を検証できる仕様とテストが揃う。 |
| P2 | 副作用なし Pure Rules Engine | 90% | 進行中 | `state.py`・`actions.py`・`transition.py` 実装済み。`test_compound_effects.py`（11テスト）を追加し開幕制限・複合効果をカバー。全52テスト PASS。 | 既存対話実装との網羅照合（残り）。 | 同じ `state, joint_action` から常に同じ `Transition` が返り、代表ルールがテストで固定される。 |
| P3 | 合法手生成・有限ストック | 75% | 進行中 | TP/NTP の合法手生成あり。ストックは `frozenset` で重複不可。既存 `get_valid_skills` との代表状態照合テストあり。 | Choice/All の後出し性や順序表現の妥当性検証。NTP reaction 側の照合/仕様固定。 | 状態依存の行動列挙が既存仕様と一致し、ストック上限が常に守られる。 |
| P4 | Mirror/Reversi 対応 | 85% | 進行中 | `RulesConfig(enable_mirror, enable_reversi)` と代表的なミラー/リバーシ遷移テストあり。`--all-configs` で 4構成 batch report 生成 CLI 追加済み。`test_four_configs.py` で 4構成 regression suite（16テスト）追加。 | 既存対話実装側への反映。 | Mirror/Reversi の ON/OFF 4構成で同じテストスイートが通る。 |
| P5 | Complete-lite exact solver | 85% | 進行中 | `state_space.py` に `enumerate_reachable_states`・`value_iteration` 実装。`--enumerate`・`--gamma-sweep` CLI 追加。gamma=0.990〜0.9995 感度分析完了（V(init)≈0.072、26 iterations 収束）。 | 支配行動除去（P8 action mask として後回し）。 | 初期局面と代表局面の均衡方策・価値が再現可能に出力される。 |
| P6 | レポート・可視化 | 90% | 進行中 | 文字化け修正・`--all-configs`・`--gamma-sweep`・`--enumerate` オプション追加。サニティ CSV に lock/time 列追加。12シナリオ対応。 | 戦術コメント列の定量化（あれば）。 | 人間が局面価値、混合方策、定性コメントを読めるレポートになる。 |
| P7 | 終盤表・戦術表 | 65% | 進行中 | `locked_flash`、`endgame_number`、`charge_number`、`quick_followup`、`endgame_me_one_opp_two`、`endgame_me_two_opp_one`、`stock_guard_flash`、`time_active`、`cement_on_me` など12シナリオ生成済み（results/complete_lite_v2/）。 | 4構成での同一シナリオ比較、`.ini` 戦術解説との定量的整合確認（ロック+セメント+フラッシュなどのコンボシナリオ追加）。 | 探索の葉評価と教師データに使える局面表が揃う。 |
| P8 | 全 Complete 環境 | 95% | 進行中 | `complete_rl/env.py`・`obs.py`・`__init__.py` 実装済み。4構成対応 CompleteEnv（Gymnasium 1.x 互換）。action mask（MaskablePPO 互換）、OBS_SIZE=107 の観測エンコーディング、ランダム/カスタム対戦相手対応。NTP 乱数を `reset(seed=...)` に連動。Gymnasium `check_env` PASS。24テスト PASS。 | 長時間学習時の wrapper/VecEnv 運用整理。 | Gymnasium 互換で 4構成を切り替えられる。 |
| P9 | 学習 AI | 70% | 進行中 | `complete_rl/maskable_ppo.py` に MaskablePPO 自己対戦 baseline CLI・評価・保存処理を追加。smoke/quick/standard preset、`--seeds` 複数 seed 評価、`--all-configs` 4構成 batch 評価、`--eval-model` 保存済みモデル評価、`--eval-dir` 評価表生成、`--ntp-policy`/`--ntp-policies` 名前付きNTP方策評価、`--reward-mode terminal|material`、`mixed_basic` / `weighted_none_counter` NTP training を追加。`complete_rl/bc_pretrain.py` に価値反復から behavioral cloning データセット生成（`generate_bc_dataset`）とアクター事前学習（`bc_pretrain`）を実装し、`--bc-pretrain/--bc-max-states/--bc-epochs/--bc-lr` CLI フラグで利用可能。`results/maskable_ppo_all_configs_quick_s0_eval/` に4構成×NTP方策の評価表、material shaping pilot 3種を生成。 | BC事前学習の本格 pilot（none/counter 弱点の改善確認）、standard の長時間学習、4構成 quick の複数 seed 化、NFSP / Deep CFR / Expert Iteration / R-NaD の実装/比較。 | 勝率だけでなく exploitability/NashConv で改善を追える。 |
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
- [x] 各スキルの golden test を網羅する。（開幕制限、Guard、Charge×2、Copy+数字、Lock を test_compound_effects.py に追加済み）
- [x] CLI/HTML レポート内の日本語文字化けを修正する。（reports.py main() に sys.stdout.reconfigure(utf-8) 追加済み）

### Complete-lite exact solver

- [x] Mirror OFF / Reversi OFF の基本解析が動く。
- [x] depth 1 の `initial`、`locked_flash`、`stock_choice`、`guarded`、`endgame_number`、`charge_number`、`quick_followup` を出力する。
- [x] depth 2 の `initial`、`locked_flash`、`endgame_number` を出力する。
- [x] 状態全列挙を実装する。（state_space.py: enumerate_reachable_states）
- [x] 割引価値反復を実装する。（state_space.py: value_iteration）
- [x] `gamma=0.995〜0.9995` の感度分析を出す。（--gamma-sweep CLI、results/gamma_sweep.csv 生成済み）
- [x] 状態数、収束回数、収束誤差をレポートに出す。（gamma_sweep.csv に states/iterations/max_delta 列あり）
- [ ] 支配行動除去または行動枝刈りを入れる。（P8以降で action mask として実装予定、暫定後回し）
- [x] `.ini` 戦術解説との整合チェックをレポート化する。（P7 シナリオ設計に活用。endgame_me_one_opp_two=0.54、time_active=0.00 など戦術解説と整合確認済み）

### Full Complete への拡張

- [x] Mirror ON / Reversi OFF の regression suite を作る。（test_four_configs.py で全4構成をカバー）
- [x] Mirror OFF / Reversi ON の regression suite を作る。（同上）
- [x] Mirror ON / Reversi ON の regression suite を作る。（同上）
- [x] 4構成すべてで同一シナリオレポートを生成する。（--all-configs CLI 追加済み）
- [x] Gymnasium 互換環境を作る。（complete_rl/env.py: CompleteEnv）
- [x] 行動マスクを実装する。（action_masks() → MaskablePPO 互換 bool array）
- [x] 観測設計を固定する。（complete_rl/obs.py: OBS_SIZE=107 の float32 ベクトル）
- [x] MaskablePPO の自己対戦 baseline を作る。（complete_rl/maskable_ppo.py、smoke 学習成果物あり）
- [x] MaskablePPO の本格学習 preset と複数 seed 評価を作る。（smoke/quick/standard、`--seeds`、summary.csv/json）
- [x] MaskablePPO quick preset を複数 seed で実走する。（seed 0/1、20k timesteps、random NTP 評価は 40/40 wins）
- [x] MaskablePPO の4構成 batch 評価 CLI を作る。（`--all-configs`、構成別モデル、all_configs_summary.csv/json）
- [x] MaskablePPO quick preset の4構成比較を実走する。（seed 0、20k timesteps、random NTP 100 episode 追試あり）
- [x] ランダム以外の NTP 評価相手を追加する。（`none`、`counter_first`、`block_first`、`mirror_first`、`--eval-model`）
- [x] 非ランダムNTP方策に対する体系的評価表を生成する。（`--eval-dir`、`--ntp-policies`、evaluation_summary.csv/json）
- [x] 打ち切り対策として任意の material reward shaping を追加し、pilot 学習する。（`--reward-mode material`、none相手の打ち切り解消を確認）
- [x] mixed/weighted NTP training を追加して pilot 学習する。（`mixed_basic`、`weighted_none_counter`、random/blockには強いがnone/counterに課題）
- [ ] NFSP / Deep CFR / Expert Iteration / R-NaD の候補を比較する。
- [ ] exploitability / NashConv / exact subgame KL を評価指標として実装する。
- [ ] 探索統合を行う。
- [ ] 4構成リーグ評価を行う。

## 確認済みコマンド

```powershell
python -m unittest discover complete_solver/tests
```
結果: 68 tests OK（complete_solver/tests）、35 tests OK（complete_rl/tests）、計103テスト PASS。作業ディレクトリは `Complete/`。

```powershell
python -m complete_solver.reports --gamma-sweep --max-states 500 --vi-epsilon 1e-6 --output results/gamma_sweep.csv
```
結果: gamma=0.990〜0.9995 の5点すべて 26 iterations で収束。V(init) = 0.0702〜0.0727（感度小）。

```powershell
python -m complete_solver.reports --all-scenarios --depth 1 --output results/complete_lite_v2
```
結果: 12シナリオ（initial〜cement_on_me）の CSV/HTML 生成済み。

```powershell
python -c "from gymnasium.utils.env_checker import check_env; from complete_rl import CompleteEnv; check_env(CompleteEnv(), skip_render_check=True); print('check_env OK')"
```
結果: check_env OK。NTP 乱数を `reset(seed=...)` に連動する修正を確認。

```powershell
python -m complete_rl.maskable_ppo --timesteps 8 --n-steps 8 --batch-size 4 --n-epochs 1 --eval-episodes 1 --max-steps 20 --output-dir results/maskable_ppo_smoke --quiet
```
結果: smoke 学習 OK。`results/maskable_ppo_smoke/maskable_ppo_complete.zip` と `metrics.json` を生成。評価は 1 episode / reward 0.0 / truncation 1（動作確認用で強さ評価ではない）。

```powershell
python -m complete_rl.maskable_ppo --preset smoke --seeds 0,1 --output-dir results/maskable_ppo_multiseed_smoke --quiet
```
結果: 複数 seed smoke 学習 OK。seed 0/1 のモデル、`summary.csv`、`summary.json` を生成。aggregate は mean_reward=0.5、mean_steps=16.0、wins=1、truncations=1（動作確認用で強さ評価ではない）。

```powershell
python -m complete_rl.maskable_ppo --preset quick --seeds 0,1 --output-dir results/maskable_ppo_quick_s0_s1 --quiet
```
結果: quick 学習 OK。20,000 timesteps × 2 seeds。20 episode 評価では seed 0/1 とも 20勝/0敗、aggregate mean_reward=1.0、mean_steps=6.95。`results/maskable_ppo_quick_s0_s1/summary.csv` と `summary.json` を生成。
同じコマンドの再実行時は既存の `seed_0/seed_1` を `reused_existing=True` として再利用する。再学習したい場合は `--force` を付ける。

```powershell
python -c "from sb3_contrib import MaskablePPO; from complete_rl.maskable_ppo import evaluate_model; m=MaskablePPO.load('results/maskable_ppo_quick_s0_s1/seed_0/maskable_ppo_complete.zip'); print(evaluate_model(m, episodes=100, seed=200000, max_steps=500).to_dict())"
python -c "from sb3_contrib import MaskablePPO; from complete_rl.maskable_ppo import evaluate_model; m=MaskablePPO.load('results/maskable_ppo_quick_s0_s1/seed_1/maskable_ppo_complete.zip'); print(evaluate_model(m, episodes=100, seed=200000, max_steps=500).to_dict())"
```
結果: seed 0 は 100勝/0敗/平均5.80 steps、seed 1 は 100勝/0敗/平均6.03 steps。random NTP 評価の追試であり、強さの最終評価ではない。

```powershell
python -m complete_rl.maskable_ppo --preset smoke --all-configs --seeds 0 --output-dir results/maskable_ppo_all_configs_smoke --quiet
```
結果: 4構成 batch smoke OK。各構成の seed_0 モデル、`all_configs_summary.csv`、`all_configs_summary.json` を生成。動作確認用で強さ評価ではない。

```powershell
python -m complete_rl.maskable_ppo --preset quick --all-configs --seeds 0 --output-dir results/maskable_ppo_all_configs_quick_s0 --quiet
```
結果: 4構成 quick 学習 OK。20,000 timesteps × 4構成 × seed 0。20 episode 評価では4構成すべて20勝/0敗。平均 steps は Mirror OFF/Reversi OFF=6.35、Mirror ON/Reversi OFF=6.10、Mirror OFF/Reversi ON=6.70、Mirror ON/Reversi ON=6.10。`results/maskable_ppo_all_configs_quick_s0/all_configs_summary.csv` と `all_configs_summary.json` を生成。

```powershell
python -c "from sb3_contrib import MaskablePPO; from complete_solver import RulesConfig; from complete_rl.maskable_ppo import evaluate_model; m=MaskablePPO.load('results/maskable_ppo_all_configs_quick_s0/mirror_off_reversi_off/seed_0/maskable_ppo_complete.zip'); print(evaluate_model(m, RulesConfig(False, False), episodes=100, seed=300000, max_steps=500).to_dict())"
python -c "from sb3_contrib import MaskablePPO; from complete_solver import RulesConfig; from complete_rl.maskable_ppo import evaluate_model; m=MaskablePPO.load('results/maskable_ppo_all_configs_quick_s0/mirror_on_reversi_off/seed_0/maskable_ppo_complete.zip'); print(evaluate_model(m, RulesConfig(True, False), episodes=100, seed=300000, max_steps=500).to_dict())"
python -c "from sb3_contrib import MaskablePPO; from complete_solver import RulesConfig; from complete_rl.maskable_ppo import evaluate_model; m=MaskablePPO.load('results/maskable_ppo_all_configs_quick_s0/mirror_off_reversi_on/seed_0/maskable_ppo_complete.zip'); print(evaluate_model(m, RulesConfig(False, True), episodes=100, seed=300000, max_steps=500).to_dict())"
python -c "from sb3_contrib import MaskablePPO; from complete_solver import RulesConfig; from complete_rl.maskable_ppo import evaluate_model; m=MaskablePPO.load('results/maskable_ppo_all_configs_quick_s0/mirror_on_reversi_on/seed_0/maskable_ppo_complete.zip'); print(evaluate_model(m, RulesConfig(True, True), episodes=100, seed=300000, max_steps=500).to_dict())"
```
結果: Mirror OFF/Reversi OFF は 100勝/0敗/平均6.18 steps、Mirror ON/Reversi OFF は 90勝/10敗/平均6.18 steps、Mirror OFF/Reversi ON は 100勝/0敗/平均6.37 steps、Mirror ON/Reversi ON は 100勝/0敗/平均5.44 steps。random NTP 評価の追試であり、強さの最終評価ではない。

```powershell
python -m complete_rl.maskable_ppo --eval-model results/maskable_ppo_quick_s0_s1/seed_0/maskable_ppo_complete.zip --eval-episodes 20 --seed 400000 --ntp-policy none --quiet
python -m complete_rl.maskable_ppo --eval-model results/maskable_ppo_quick_s0_s1/seed_0/maskable_ppo_complete.zip --eval-episodes 20 --seed 400000 --ntp-policy counter_first --quiet
python -m complete_rl.maskable_ppo --eval-model results/maskable_ppo_quick_s0_s1/seed_0/maskable_ppo_complete.zip --eval-episodes 20 --seed 400000 --ntp-policy block_first --quiet
python -m complete_rl.maskable_ppo --eval-model results/maskable_ppo_all_configs_quick_s0/mirror_on_reversi_off/seed_0/maskable_ppo_complete.zip --mirror --eval-episodes 20 --seed 400000 --ntp-policy mirror_first --quiet
```
結果: quick seed 0 / Mirror OFF/Reversi OFF モデルは `none` で 0勝/0敗/20打ち切り、`counter_first` で20勝/0敗/平均3.0 steps、`block_first` で20勝/0敗/平均4.0 steps。Mirror ON/Reversi OFF モデルの `mirror_first` 評価は0勝/0敗/20打ち切り。非ランダムNTP評価の入口確認であり、体系的な強さ評価は次タスク。

```powershell
python -m complete_rl.maskable_ppo --eval-dir results/maskable_ppo_all_configs_quick_s0 --all-configs --eval-output results/maskable_ppo_all_configs_quick_s0_eval --eval-episodes 20 --seed 400000 --ntp-policies random,none,counter_first,block_first,mirror_first --quiet
```
結果: 4構成 × 5 NTP方策の評価表生成 OK。`evaluation_summary.csv` と `evaluation_summary.json` を生成。主な結果は、全構成で `none` が20打ち切り、Mirror ON/Reversi OFF では `counter_first` と `mirror_first` も20打ち切り、その他の多くの組み合わせは20勝/0敗。random 評価は Mirror ON/Reversi OFF のみ18勝/2敗、他3構成は20勝/0敗。

```powershell
python -m complete_rl.maskable_ppo --timesteps 5000 --n-steps 256 --batch-size 64 --n-epochs 4 --eval-episodes 20 --max-steps 200 --reward-mode material --ntp-policy none --output-dir results/maskable_ppo_material_none_t5k --quiet --force
```
結果: material shaping pilot 学習 OK。none 相手で20勝/0敗/平均8.0 steps。`results/maskable_ppo_material_none_t5k/` にモデルと metrics を生成。

```powershell
python -m complete_rl.maskable_ppo --eval-model results/maskable_ppo_material_none_t5k/maskable_ppo_complete.zip --eval-episodes 50 --max-steps 200 --seed 500000 --ntp-policy none --quiet
python -m complete_rl.maskable_ppo --eval-model results/maskable_ppo_material_none_t5k/maskable_ppo_complete.zip --eval-episodes 50 --max-steps 200 --seed 500000 --ntp-policy random --quiet
python -m complete_rl.maskable_ppo --eval-model results/maskable_ppo_material_none_t5k/maskable_ppo_complete.zip --eval-episodes 50 --max-steps 200 --seed 500000 --ntp-policy counter_first --quiet
```
結果: terminal評価で none は50勝/0敗/平均8.0 steps、random は31勝/19敗/平均5.18 steps、counter_first は0勝/50敗/平均4.0 steps。打ち切り解消には効いたが単一NTP方策への過適応が強い。

```powershell
python -m complete_rl.maskable_ppo --timesteps 5000 --n-steps 256 --batch-size 64 --n-epochs 4 --eval-episodes 20 --max-steps 200 --reward-mode material --ntp-policy mixed_basic --output-dir results/maskable_ppo_material_mixed_t5k --quiet --force
```
結果: mixed_basic + material pilot 学習 OK。混合評価では19勝/1敗/平均4.75 steps。

```powershell
python -m complete_rl.maskable_ppo --eval-model results/maskable_ppo_material_mixed_t5k/maskable_ppo_complete.zip --eval-episodes 50 --max-steps 200 --seed 600000 --ntp-policy none --quiet
python -m complete_rl.maskable_ppo --eval-model results/maskable_ppo_material_mixed_t5k/maskable_ppo_complete.zip --eval-episodes 50 --max-steps 200 --seed 600000 --ntp-policy random --quiet
python -m complete_rl.maskable_ppo --eval-model results/maskable_ppo_material_mixed_t5k/maskable_ppo_complete.zip --eval-episodes 50 --max-steps 200 --seed 600000 --ntp-policy counter_first --quiet
python -m complete_rl.maskable_ppo --eval-model results/maskable_ppo_material_mixed_t5k/maskable_ppo_complete.zip --eval-episodes 50 --max-steps 200 --seed 600000 --ntp-policy block_first --quiet
```
結果: none は0勝/0敗/50打ち切り、random は48勝/2敗/平均6.46 steps、counter_first は0勝/50敗/平均4.0 steps、block_first は50勝/0敗/平均4.0 steps。単純な均等混合だけでは none/counter の弱点は解消しきれない。

```powershell
python -m complete_rl.maskable_ppo --timesteps 5000 --n-steps 256 --batch-size 64 --n-epochs 4 --eval-episodes 20 --max-steps 200 --reward-mode material --ntp-policy weighted_none_counter --output-dir results/maskable_ppo_material_weighted_t5k --quiet --force
```
結果: weighted_none_counter + material pilot 学習 OK。混合評価では18勝/2敗/平均5.8 steps。

```powershell
python -m complete_rl.maskable_ppo --eval-model results/maskable_ppo_material_weighted_t5k/maskable_ppo_complete.zip --eval-episodes 50 --max-steps 200 --seed 700000 --ntp-policy none --quiet
python -m complete_rl.maskable_ppo --eval-model results/maskable_ppo_material_weighted_t5k/maskable_ppo_complete.zip --eval-episodes 50 --max-steps 200 --seed 700000 --ntp-policy random --quiet
python -m complete_rl.maskable_ppo --eval-model results/maskable_ppo_material_weighted_t5k/maskable_ppo_complete.zip --eval-episodes 50 --max-steps 200 --seed 700000 --ntp-policy counter_first --quiet
python -m complete_rl.maskable_ppo --eval-model results/maskable_ppo_material_weighted_t5k/maskable_ppo_complete.zip --eval-episodes 50 --max-steps 200 --seed 700000 --ntp-policy block_first --quiet
```
結果: none は0勝/0敗/50打ち切り、random は42勝/8敗/平均6.28 steps、counter_first は0勝/0敗/50打ち切り、block_first は50勝/0敗/平均4.0 steps。重み付き混合でも none/counter の弱点は残り、教師データや局面探索に基づく改善が必要。

```powershell
python -m complete_rl.maskable_ppo --timesteps 8 --n-steps 8 --batch-size 4 --n-epochs 1 --eval-episodes 1 --max-steps 20 --bc-pretrain --bc-max-states 30 --bc-epochs 2 --output-dir results/maskable_ppo_bc_smoke --force --quiet
```
結果: BC smoke OK。`bc_pretrain.py`：価値反復（30 states）→ BC 2 epoch → MaskablePPO 8 steps の流れが正常動作。113テスト（complete_solver 68 + complete_rl 45）全 PASS。

`pytest` は現在の Python 環境に未導入のため、標準の `unittest` で確認している。

## 次にやる順番

1. ~~CLI/HTML レポート内の日本語文字化けを修正する。~~ ✅ 完了
2. ~~Mirror reaction の残りスキルと、開幕制限/複合効果の golden test を追加する。~~ ✅ 完了
3. ~~4構成の batch report を生成できる CLI オプションまたはスクリプトを追加する。~~ ✅ 完了
4. ~~状態列挙と価値反復に進む。~~ ✅ 完了
5. ~~gamma 感度分析と収束ログ出力。~~ ✅ 完了
6. ~~P7 体系的終盤表の追加（.ini 参照）。~~ ✅ 完了（12シナリオ）
7. ~~4構成 regression suite を作る。~~ ✅ 完了（test_four_configs.py：16テスト、0.5秒）
8. ~~P8 Gymnasium 互換環境を作る。~~ ✅ 完了（complete_rl/：env.py・obs.py・24テスト、check_env OK）
9. ~~P9 MaskablePPO 自己対戦 baseline を作る。~~ ✅ 完了（complete_rl/maskable_ppo.py、smoke 学習 OK）
10. ~~P9 本格学習設定と複数 seed 評価を作る。~~ ✅ 完了（preset、`--seeds`、summary 出力）
11. ~~P9 quick preset の長時間学習を回し、複数 seed の安定評価を取る。~~ ✅ 完了（seed 0/1、100 episode 追試）
12. ~~P9 4構成 batch 評価 CLI を作る。~~ ✅ 完了（`--all-configs`、smoke 確認）
13. ~~P9 quick の4構成比較を実走する。~~ ✅ 完了（seed 0、100 episode 追試）
14. ~~ランダム以外の評価相手を追加する。~~ ✅ 完了（named NTP policies、`--eval-model`）
15. ~~非ランダムNTP方策に対する体系的評価表を生成する。~~ ✅ 完了（`--eval-dir`、4構成×5方策）
16. ~~打ち切りが多い `none` への対策として material reward shaping を試す。~~ ✅ 完了（noneには有効、counter_firstには過適応）
17. ~~mixed NTP training を作り、none/counter/block/random への過適応を減らす。~~ ✅ 部分完了（random/block改善、none/counter課題残り）
18. ~~weighted mixed NTP training を作り、none/counter の弱点を潰す。~~ ✅ 試行完了（弱点は残存）
19. ~~exact solver / scenario policy を教師にした imitation warm-start または探索統合を検討する。~~ ✅ 完了（`bc_pretrain.py`：`generate_bc_dataset` + `bc_pretrain`、`--bc-pretrain` CLI、10テスト追加、113テスト PASS）
20. **BC事前学習 pilot を実走して none/counter 弱点の改善を確認する。** ← **次**
21. **P9 4構成 quick を複数 seed 化する、または standard preset の長時間学習へ進む。**
22. **NFSP / Deep CFR / Expert Iteration / R-NaD の候補比較に進む。**
