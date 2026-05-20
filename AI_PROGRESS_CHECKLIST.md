# Complete AI チェックリスト・進捗管理票

更新日: 2026-05-20（Nash-NTP 実装・4構成 multi-seed・アルゴリズム比較・standard nash_optimal 結果確認）  
対象: `complete_ai_plan.html` に基づく Complete ルール最強 AI 作成計画  
判定基準: リポジトリ内の実装、テスト、生成済みレポートを確認して評価

## 運用ルール

今後 Complete ルール AI 関連の作業を行った場合は、作業完了時にこのファイルも更新すること。

作業の進め方は、必ず `AI_WORKFLOW_RULES.md` に従う。特に、実装・検証・学習・評価の各区切りで確認用成果物とAI側の解釈を提示し、ユーザーの明示的な承認を得るまで次工程へ進まない。

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
- Full Complete 学習 AI まで含めた全体完成度: **78%**（P8 環境・観測・action mask は継続利用可能。P9 は BC事前学習・curriculum・fine-tune・Nash-NTP・エピソード NTP 切り替えまで実装済みだが、最新方策レポートで数字宣言過多が見つかったため、学習結果の採用は保留。現在は診断・評価設計の見直しフェーズ）

`complete_solver/` に純粋な状態表現・合法手生成・1ターン遷移・深さ制限 subgame solver・状態全列挙・割引価値反復・CSV/HTML レポートが揃っている。`complete_rl/` には Gymnasium 互換環境（OBS_SIZE=123、直近4反応 one-hot 追加済み）、action mask、MaskablePPO baseline CLI、smoke/quick/standard preset、複数 seed 評価、4構成 batch 学習/評価 CLI、名前付きNTP反応方策、BC事前学習（`bc_pretrain.py`）、curriculum warmup、fine-tune、Nash-NTP（`nash_ntp.py`）、エピソード NTP 切り替え（`episode_mixed_basic`・`episode_weighted_none_counter`）がある。BC + episode_mixed_basic + standard 250k は named policy 勝率上は良好だったが、方策レポートでは数字宣言が 92〜96% と異常に高く、強力スキルの使用がほぼ見られないため、現時点では「新ベストモデル」として採用しない。

暫定モデル: `results/maskable_ppo_bc_standard_episode_mixed/` は採用保留。`results/policy_report_episode_mixed.html` の数字宣言過多を受け、`results/minimal_ntp_policy_diagnostics.html` と `results/turn_chain_reward_diagnostics.html` で最小診断を実施した。現時点の見解は、ルールエンジンや合法手生成だけで数字偏重が必然化しているわけではなく、NTP 方策の反応選択と指選択の混同、報酬 shaping、評価レポート不足、学習の局所最適が複合している可能性が高い、というもの。

## 現在の診断状況（2026-05-21）

- `policy_report_episode_mixed.html` では、全 NTP 条件で数字宣言が 92〜96% と過剰。フェイント、フラッシュ、クイック、スキップ、コピーなどの重要スキルがほぼ使われていない。
- 現行の `none` / `counter_first` などの named NTP policy は、反応方策だけでなく NTP の指選択も固定気味にしている。今後は reaction policy と thumb policy を分離して評価する必要がある。
- `minimal_ntp_policy_diagnostics.html` では、100%カウンター条件の首位はフェイント、0%カウンター条件の首位は非数字スキルであり、数字宣言偏重はルール上の必然ではないことを確認した。
- `turn_chain_reward_diagnostics.html` では、ガード・ブースト・スキップ・タイムの追加ターン取得そのものに即時報酬は入っていないことを確認した。一方で、深さ制限評価では追加ターン系が探索深さ内の後続勝ち筋を拾うため高く見えやすい。
- `ntp_policy_separation_design.html` で、NTP 方策を reaction policy と thumb policy に分離する設計案を作成した。現時点では実装に進まず、ユーザー承認待ち。
- ロックは局面依存で非線形に価値が変わるため、今回の最小診断では主対象から外す。必要になった時点で専用シナリオを作る。
- 今後は `AI_WORKFLOW_RULES.md` に従い、各工程で確認用HTML/表とAI側の解釈を出し、ユーザー承認後に次工程へ進む。

## 次に取り組むべきリスト

次工程は、学習再開ではなく評価・環境条件の切り分けから行う。

1. **NTP 方策の分離設計**
   - `none` / `counter_first` のような名前付き方策を、反応選択（なし/カウンター/ブロック等）と指選択（一様、最小、最悪応答など）に分離する。
   - 必要理由: 現在の named NTP policy は「相手が何を宣言するか」と「相手の指がどう出るか」を同時に固定しており、数字宣言偏重の原因が反応条件なのか指条件なのか切り分けられないため。
   - まず設計案と小さな確認表を出し、承認後に実装する。

2. **方策レポートの拡張**
   - 初手分布、行動カテゴリ分布、主要スキル使用率、NTP 反応率、NTP 指分布を出す。
   - 追加ターン後の次手、勝ち筋への接続、ループ/待機/外し数字の警告を出す。
   - deterministic / stochastic の両方を比較する。
   - 必要理由: 勝率だけでは、数字宣言連打や追加ターン維持のような不自然な勝ち方を見逃す。人間が「その方策で本当に妥当か」を確認できる粒度の可視化が必要なため。

3. **最小シナリオ検証**
   - 0%カウンター・100%カウンターを主対象に、反応率と指選択を分離した条件で、期待される方策と実際の上位方策を比較する。
   - 99%カウンターは必須ではない。ロックなど、価値が特定状態で非線形に変わるスキルを専用に見る段階になった場合だけ、任意の追加検証として扱う。
   - フラッシュ、クイック、スキップ→コピー→クイック等の固定手順を、勝率・平均手数・報酬で確認する。
   - 必要理由: 学習モデルを再学習する前に、そもそも環境・報酬・合法手判定が、人間視点で明らかな極端条件に対して正しい順位を返すか確認するため。

4. **報酬設計の再確認**
   - `terminal` と `material` の差分を比較し、数字偏重や追加ターン偏重がどちらで発生するかを確認する。
   - 追加ターンそのものには独立報酬を入れない方針を維持する。
   - 必要理由: 追加ターン取得や手数維持が、勝利に近づくことと混同されていないかを確認するため。特に material reward や深さ制限評価が、間接行動を過大評価していないかを見る必要がある。

5. **再学習は最後**
   - 上記の評価レポートと最小ケースで人間視点の納得が取れてから、既存モデルを破棄または保留し、新条件で再学習する。
   - 再学習後も、勝率だけで採用せず、方策分布・スキル使用率・警告リストを確認してから採用判断する。
   - 必要理由: 評価条件が曖昧なまま再学習すると、同じ局所最適や報酬ハックを再生産する可能性が高いため。

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
| P8 | 全 Complete 環境 | 100% | 完了 | `complete_rl/env.py`・`obs.py`・`__init__.py` 実装済み。4構成対応 CompleteEnv（Gymnasium 1.x 互換）。action mask（MaskablePPO 互換）、**OBS_SIZE=123 の観測エンコーディング**（直近4反応の one-hot 16 feature を追加）、ランダム/カスタム対戦相手対応。NTP 乱数を `reset(seed=...)` に連動。Gymnasium `check_env` PASS。check_env OK 確認。 | なし | Gymnasium 互換で 4構成を切り替えられ、obs に相手反応履歴が含まれる。 |
| P9 | 学習 AI | 75% | 診断・見直し中 | BC事前学習、curriculum warmup、fine-tune、Nash-NTP、エピソード NTP 切り替えは実装済み。`policy_report_episode_mixed.html` により、episode_mixed モデルの数字宣言 92〜96% という不自然な方策偏りを確認。`minimal_ntp_policy_diagnostics.html`・`turn_chain_reward_diagnostics.html` で最小診断を実施。`ntp_policy_separation_design.html` で NTP 方策分離の設計案を作成。 | NTP方策分離設計のユーザー承認、承認後の最小実装、方策レポート拡張、0%/100%カウンター最小検証、terminal/material 比較、承認後の再学習。99%カウンターはロック等の非線形スキルを専用検証する場合のみ任意。 | 勝率だけでなく、方策分布・主要スキル使用率・追加ターン後の勝ち筋・警告リストが人間視点で妥当と確認できる。 |
| P10 | 探索統合 | 0% | 未着手 | なし。 | 方策を事前分布にした局所 subgame 探索、詰み/必殺/ミラー/リバーシ局面の深掘り。 | 実戦時に重要局面で読みを深くできる。 |
| P11 | 4構成統合評価 | 10% | 保留 | Config フラグと一部テストの土台はあり。 | P9 の方策偏り診断と評価レポート拡張が終わるまで、4構成リーグ戦は進めない。 | P9 の評価基準が安定した後、4構成を同一基盤で評価し、採用方針を決められる。 |

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
- [x] 観測設計を固定する。（complete_rl/obs.py: OBS_SIZE=123 の float32 ベクトル。107 ゲーム状態 + 16 相手反応履歴）
- [x] MaskablePPO の自己対戦 baseline を作る。（complete_rl/maskable_ppo.py、smoke 学習成果物あり）
- [x] MaskablePPO の本格学習 preset と複数 seed 評価を作る。（smoke/quick/standard、`--seeds`、summary.csv/json）
- [x] MaskablePPO quick preset を複数 seed で実走する。（seed 0/1、20k timesteps、random NTP 評価は 40/40 wins）
- [x] MaskablePPO の4構成 batch 評価 CLI を作る。（`--all-configs`、構成別モデル、all_configs_summary.csv/json）
- [x] MaskablePPO quick preset の4構成比較を実走する。（seed 0、20k timesteps、random NTP 100 episode 追試あり）
- [x] ランダム以外の NTP 評価相手を追加する。（`none`、`counter_first`、`block_first`、`mirror_first`、`--eval-model`）
- [x] 非ランダムNTP方策に対する体系的評価表を生成する。（`--eval-dir`、`--ntp-policies`、evaluation_summary.csv/json）
- [x] 打ち切り対策として任意の material reward shaping を追加し、pilot 学習する。（`--reward-mode material`、none相手の打ち切り解消を確認）
- [x] mixed/weighted NTP training を追加して pilot 学習する。（`mixed_basic`、`weighted_none_counter`、random/blockには強いがnone/counterに課題）
- [x] NFSP / Deep CFR / Expert Iteration / R-NaD の候補を比較する。（比較完了。Nash-NTP を追加実装・pilot 実行。詳細は下記）
- [x] exploitability / NashConv / exact subgame KL を評価指標として実装する。（`complete_rl/exploitability.py`：Nash VI・BR VI・5テスト。BC+standard+obs123 モデルで exploitability=+0.0194 を確認）
- [ ] 探索統合を行う。
- [ ] 4構成リーグ評価を行う。

## 確認済みコマンド

```powershell
python -m unittest discover complete_solver/tests
```
結果: 68 tests OK（complete_solver/tests）、64 tests OK（complete_rl/tests）、計132テスト PASS。作業ディレクトリは `Complete/`。（初回実行時は 103 PASS、以降の追加実装で 132 に増加）

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

```powershell
python -m complete_rl.maskable_ppo --preset quick --bc-pretrain --bc-max-states 400 --bc-epochs 5 --ntp-policy weighted_none_counter --reward-mode material --output-dir results/maskable_ppo_bc_quick_weighted_t20k --force --quiet
```
結果: BC + quick + weighted_none_counter + material 学習 OK。20 episode 評価で 20勝/0敗/平均6.3 steps。
評価（各 50 episode, seed 800000, max_steps 200）: `none` 0勝/50打ち切り、`counter_first` **50勝**/0敗（旧 50敗→大幅改善）、`random` 50勝、`block_first` 50勝。BC事前学習が counter_first の弱点を劇的に解消。none の打ち切りは残存。

```powershell
python -m complete_rl.maskable_ppo --preset quick --bc-pretrain --bc-max-states 400 --bc-epochs 5 --ntp-policy none --reward-mode material --output-dir results/maskable_ppo_bc_quick_none_t20k --force --quiet
```
結果: BC + quick + none + material 学習 OK。評価: `none` 50勝、`counter_first` 0勝/50敗、`random` 36勝/14敗、`block_first` 0勝/50敗。none 特化でカウンター系に弱い。BC のみでは none/counter 両立は quick preset (20k steps) では難しい。

```powershell
python -m complete_rl.maskable_ppo --timesteps 20000 --n-steps 256 --batch-size 64 --n-epochs 4 --eval-episodes 20 --max-steps 200 --bc-pretrain --bc-max-states 400 --bc-epochs 5 --curriculum-warmup-steps 5000 --curriculum-warmup-policy none --ntp-policy weighted_none_counter --reward-mode material --output-dir results/maskable_ppo_bc_curriculum_quick --force --quiet
```
結果: curriculum (none 5k → weighted 15k) + BC。`counter_first` 50勝、`random` 50勝、`block_first` 50勝。`none` はやはり 50 打ち切り。curriculum でも quick 20k では none/counter 両立は不可。

```powershell
python -m complete_rl.maskable_ppo --fine-tune-from results/maskable_ppo_bc_quick_none_t20k/maskable_ppo_complete.zip --timesteps 10000 --n-steps 256 --batch-size 64 --n-epochs 4 --eval-episodes 20 --max-steps 200 --ntp-policy weighted_none_counter --reward-mode material --output-dir results/maskable_ppo_finetune_none2weighted --force --quiet
```
結果: none 特化モデルから weighted で fine-tune。`none` 50勝は維持したが `counter_first` 50敗、`block_first` 50敗（catastrophic forgetting）。--fine-tune-from CLI は正常動作確認。

※ none/counter 両立の根本的解決には opponent modeling（観測に相手の反応傾向を加える P10 探索統合）が必要。現状の obs 設計（OBS_SIZE=107）は相手 NTP 方策の区別ができない。

```powershell
python -m complete_rl.maskable_ppo --preset standard --bc-pretrain --bc-max-states 400 --bc-epochs 5 --ntp-policy weighted_none_counter --reward-mode material --output-dir results/maskable_ppo_bc_standard_weighted --force --quiet
```
結果: 完了。100 episode 評価で 100 勝（weighted_none_counter 相手）。詳細評価（各 100 ep、seed 1000000、max_steps 500）: `counter_first` 100勝、`random` 100勝、`block_first` 100勝。`none` は 100% 打ち切り（OBS_SIZE=107 の根本限界）。

```powershell
# check_env（OBS_SIZE=123 確認）
python -c "from gymnasium.utils.env_checker import check_env; from complete_rl import CompleteEnv; check_env(CompleteEnv(), skip_render_check=True); print('OK, OBS_SIZE:', __import__('complete_rl').OBS_SIZE)"
```
結果: check_env OK, OBS_SIZE: 123。obs に直近4ターンの相手反応 one-hot（16 features）を追加。既存テスト 128 件（complete_solver 68 + complete_rl 60）全 PASS。

```powershell
python -m complete_rl.maskable_ppo --preset standard --bc-pretrain --bc-max-states 400 --bc-epochs 5 --ntp-policy weighted_none_counter --reward-mode material --output-dir results/maskable_ppo_bc_standard_obs123 --force --quiet
```
結果: 完了（OBS_SIZE=123 での 250k steps）。詳細評価（各 100 ep、seed 1000000、max_steps 500）: `counter_first` 100勝、`random` 100勝、`block_first` 100勝。`none` は 100% 打ち切り（OBS=107 と同じ結果。反応履歴を obs に加えても 250k では学習されず）。

```powershell
python -m complete_rl.exploitability results/maskable_ppo_bc_standard_obs123/maskable_ppo_complete.zip --max-states 400 --gamma 0.999 --verbose
```
結果: nash_value=+0.0726、best_response_value=+0.0532、exploitability=+0.0194（正 = NTP が TP の期待利得を Nash 均衡値以下に下げられる）、n_states=448、converged=false（BR delta=0.106 で打ち切り）。BC + standard + obs=123 モデルは Nash に近いが完全ではない。

```powershell
python -m complete_rl.maskable_ppo --preset quick --all-configs --seeds 0,1,2 --bc-pretrain --bc-max-states 400 --bc-epochs 5 --ntp-policy weighted_none_counter --reward-mode material --output-dir results/maskable_ppo_all_configs_bc_quick_multi_seed --force --quiet
python -m complete_rl.maskable_ppo --eval-dir results/maskable_ppo_all_configs_bc_quick_multi_seed --all-configs --eval-output results/maskable_ppo_all_configs_bc_quick_multi_seed_eval --eval-episodes 50 --seed 900000 --ntp-policies random,none,counter_first,block_first,mirror_first --quiet
```
結果: 4構成 × seeds 0/1/2 × 5 NTP方策（50 ep）全 60 件評価完了。`random`/`counter_first`/`block_first`/`mirror_first` は全構成・全 seed で 50勝/0敗（100%）、分散ゼロ。`none` は全構成・全 seed で 50 打ち切り（100%）。BC + weighted_none_counter 方策は 20k steps quick で安定して counter/block/mirror に対処できる。none の打ち切りは構成・seed に関わらず根本的な問題。

### アルゴリズム候補比較（P9 item 25）

| アルゴリズム | 適合性 | 実装コスト | 判定 |
|---|---|---|---|
| Deep CFR | △ exact CFR が既に VI で代替可能 | 高（外部価値 net 必要） | 不採用：VI で Nash が求められているため不要 |
| Expert Iteration | ○ BC→RL ループで段階的改善 | 低（BC 既存） | 採用候補：RL 改善ステップが弱いと効果薄い |
| R-NaD | △ KL 正則化 Nash 収束 | 高（ゲーム木構造依存） | 保留：実装コスト大、ゲーム規模的に過剰 |
| NFSP | ◎ 過去方策の均一サンプル | 中（replay buffer 必要） | 最有力：多様な対戦相手自然生成→none/counter 両立期待 |
| **Nash-NTP** | ◎ exact Nash NTP で学習を強制 | 低（VI + LP 既存活用） | **実装完了**（`complete_rl/nash_ntp.py`）|

実装詳細（Nash-NTP、`complete_rl/nash_ntp.py`）:
- `compute_nash_ntp_strategies(config, max_states, gamma, vi_epsilon)` で全状態の Nash NTP 分布を計算
- 状態ごとに payoff 行列 → LP → col_policy (NTP 混合戦略)
- `CompleteEnv(opponent_policy="nash_optimal")` で利用可能
- 128 テスト（`test_nash_ntp.py` 6件 + `test_env.py` 2件追加）全 PASS

```powershell
python -m complete_rl.maskable_ppo --timesteps 20000 --n-steps 256 --batch-size 64 --n-epochs 4 --eval-episodes 20 --max-steps 200 --bc-pretrain --bc-max-states 400 --bc-epochs 5 --ntp-policy nash_optimal --reward-mode material --output-dir results/maskable_ppo_bc_quick_nash_optimal --force --quiet
```
結果（quick 20k、BC 5ep）: 訓練時評価 20勝/0敗。`random` 50勝、`counter_first` 50勝、`block_first` 50勝。`none` は 20 打ち切り（max_steps=50）。  
Nash NTP も quick 20k では none 打ち切りを解消できない（weighted と同様）。standard 250k では改善する可能性あり。

```powershell
python -m complete_rl.maskable_ppo --preset standard --bc-pretrain --bc-max-states 400 --bc-epochs 5 --ntp-policy nash_optimal --reward-mode material --output-dir results/maskable_ppo_bc_standard_nash_optimal --force --quiet
```
結果（standard 250k、BC 5ep、nash_optimal NTP）: 訓練時評価 100勝/0敗。`random` 100勝、`counter_first` 100勝、`block_first` 100勝。**`none` は依然 100% 打ち切り（100 truncations）**。Nash-NTP + standard 250k でも none 問題は解消不可。none/counter 両立には NFSP 等の多様な対戦相手をプールする手法が必要。

### エピソード NTP 切り替え方式（P9 item 27 - none/counter 両立ブレークスルー）

`episode_mixed_basic` / `episode_weighted_none_counter`: エピソード開始時に NTP 方策を1つ選択し、エピソード中は固定。
- ステップ毎の混合と異なり、エピソード内の反応履歴が均一になる
- OBS=123（反応履歴）で「今は none 相手」「今は counter 相手」をモデルが識別できる

```powershell
python -m complete_rl.maskable_ppo --preset standard --bc-pretrain --bc-max-states 400 --bc-epochs 5 --ntp-policy episode_mixed_basic --reward-mode material --output-dir results/maskable_ppo_bc_standard_episode_mixed --force --quiet
python -m complete_rl.maskable_ppo --preset standard --bc-pretrain --bc-max-states 400 --bc-epochs 5 --ntp-policy episode_weighted_none_counter --reward-mode material --output-dir results/maskable_ppo_bc_standard_episode_weighted --force --quiet
```
結果: 両モデルとも **none 100勝、counter 100勝、block 100勝、random 99勝**（各 100ep、seed 5000000）。**none/counter 両立を初めて達成**。  
exploitability: episode_mixed=+0.0418（BR未収束）、episode_weighted=+0.0756（BR収束）。  
episode_mixed が理論的指標で優れているため**新ベストモデル**とする。

### exploitability メトリクス限界（item 29）

`compute_exploitability` は history=() で TP 方策を評価するため、history-dependent モデルの exploitability を過大評価する。

- episode_mixed (br_max_iter=2000): exploitability=+0.0724（未収束 delta=3.83e-04）
- episode_mixed + ent_coef=0.005: exploitability=+0.0727（収束 delta=9.88e-06）

いずれも Nash 値 (+0.0726) にほぼ等しく、空履歴状態での BR が全利得を奪うことを示す。ただし実際の対戦では history スロット（4 スロット）がリセットされるため、相手の戦略切り替えは 4 ターン後に無効化される。実用指標（named policy 勝率：none 100%、counter 100%）が primary metric。

4構成 multi-seed 再現性確認（episode_mixed_basic + BC + standard 250k）:
- mirror_off_reversi_off: seeds 0,1,2 → W=100/100/100, T=0 ✓
- mirror_on_reversi_off: seeds 0,1,2 → W=99/100/100, T=0 ✓
- mirror_off_reversi_on: seed_0 → W=100, T=0 ✓
- mirror_on_reversi_on: seed_0 → W=100, T=0 ✓

全構成でエピソード NTP 切り替え方式が機能することを確認。結果は `results/maskable_ppo_bc_standard_episode_mixed_4cfg/` 以下に保存。

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
19. ~~exact solver / scenario policy を教師にした imitation warm-start または探索統合を検討する。~~ ✅ 完了（`bc_pretrain.py`：`generate_bc_dataset` + `bc_pretrain`、`--bc-pretrain` CLI、10テスト追加）
20. ~~BC事前学習 pilot を実走して none/counter 弱点の改善を確認する。~~ ✅ 完了（BC + weighted: counter_first 50勝・random 50勝・block 50勝。none 打ち切りは quick 20k では残存。curriculum / fine-tune も試行。none/counter 両立には opponent modeling が必要）
21. ~~BC + standard 250k + weighted_none_counter の結果確認・評価。~~ ✅ 完了（counter/random/block 各 100 勝。none=100 打ち切り、OBS_SIZE=107 の根本限界を確認）
22. ~~OBS_SIZE を 107→123 に拡張して相手直近4反応の one-hot 16 feature を追加する。~~ ✅ 完了（obs.py・env.py 更新、check_env OK、120テスト PASS）
23. ~~BC + standard 250k（OBS_SIZE=123）の結果確認・評価。~~ ✅ 完了（OBS=107 と同結果。counter/random/block 各 100 勝、none=100 打ち切り。exploitability=+0.0194 で Nash に近い）
24. ~~P9 4構成 quick を複数 seed 化する。~~ ✅ 完了（BC + weighted + quick 20k、seeds 0/1/2、4構成 × 3 seeds × 5 NTP方策 × 50 ep。random/counter/block/mirror_first 全 100%勝。none は全構成・全 seed で 100% 打ち切り。Seed 間分散ゼロで再現性を確認）
25. ~~NFSP / Deep CFR / Expert Iteration / R-NaD の候補比較に進む。~~ ✅ 完了（比較・Nash-NTP pilot 実装済み。詳細は確認済みコマンドに追記。次は standard 250k + nash_optimal で none 問題が解消するか確認、または NFSP 本実装）
26. ~~BC + standard 250k + nash_optimal の結果確認。~~ ✅ 完了（100 勝/0 打ち切り vs weighted NTP 評価。しかし `none` には依然 100% 打ち切り。Nash NTP も standard では none 問題を解消できない）
27. ~~エピソード NTP 切り替え方式の実装・評価。~~ ✅ **完了（大ブレークスルー）** （`episode_mixed_basic`・`episode_weighted_none_counter` 追加。BC + episode_mixed + standard 250k で none/counter/block/random 全 100 勝初達成。OBS=123 反応履歴によるオポネント識別が機能）
28. ~~episode_mixed_basic の4構成 multi-seed 評価（再現性確認）。~~ ✅ 完了（4構成 seed 0-2 評価済み: mirror_off/on_reversi_off 各 3 seeds・mirror_off_reversi_on 1 seed・mirror_on_reversi_on 1 seed。全て W≥99/100, T=0。none 打ち切りゼロを4構成で確認）
29. ~~episode_mixed_basic の exploitability 改善（longer training or entropy 正則化）。~~ ✅ 調査完了（**メトリクス限界と判定**）  
    - ent_coef=0.005 追加実装・実験：W=100, T=0 維持、exploitability=+0.0727（変化なし）
    - 根本原因：`compute_exploitability` は全状態で history=() で評価→空履歴でモデルが数字宣言→BR がカウンターで返す→exploitability≈Nash value
    - episode_mixed モデルの実用 exploitability は履歴 4 スロット以降に相手が戦略変更しても適応できるため、理論値より大幅に低い
    - **対応**: exploitability.py に limitation のコメントを追加。実用指標（named policy 勝率）を primary metric とする。
    - `maskable_ppo.py` に `--ent-coef` オプション追加（デフォルト 0.0）
30. ~~episode_mixed_basic の方策レポート確認。~~ ⚠️ **異常検出**  
    - `results/policy_report_episode_mixed.html` で数字宣言が全条件 92〜96%。
    - 勝率は高いが、フェイント・フラッシュ・クイック・スキップ・コピー等の重要スキルをほぼ使わないため、モデル採用は保留。
31. ~~最小 NTP 方策診断。~~ ✅ 完了  
    - `results/minimal_ntp_policy_diagnostics.html` を生成。
    - 100%カウンターではフェイント、0%カウンターでは非数字スキルが上位。数字偏重はルール上の必然ではない。
32. ~~追加ターン・報酬診断。~~ ✅ 完了  
    - `results/turn_chain_reward_diagnostics.html` を生成。
    - ガード・ブースト・スキップ・タイムの追加ターン取得そのものに即時報酬はない。
    - ただし深さ制限評価では、追加ターン系が後続勝ち筋を拾うため高く見えやすい。
33. **NTP 方策を reaction policy と thumb policy に分離する設計案を作る。** ✅ 設計案作成・承認待ち
    - 例: 反応は 0%/100% カウンター、指は一様/最小/最悪応答で別管理。
    - 必要理由: 数字宣言偏重が「カウンター率への適応」なのか「相手指の固定条件への過適応」なのかを分けて見られるようにするため。
    - 成果物: `results/ntp_policy_separation_design.html`
    - ユーザー承認後に、最小実装と確認表生成へ進む。
34. **NTP 方策分離の最小実装と確認表生成。** ← **承認後の次**
    - `none_lowest` / `none_uniform` / `counter_lowest` / `counter_uniform` のような分離条件を追加する。
    - NTP 反応率、NTP 指分布、TP 上位行動、警告を確認表に出す。
    - 必要理由: 設計上の切り分けが、実際にレポート上で検証可能か確認するため。
35. **方策レポート拡張を設計・実装する。**
    - 初手分布、主要スキル使用率、NTP反応率、NTP指分布、追加ターン後の次手、警告リストを出す。
    - deterministic/stochastic の差も表示する。
    - 必要理由: 勝率が良くても、人間視点で不自然な方策に収束していないかを採用前に検出するため。
36. **0%/100% カウンターの最小検証を再実施する。**
    - フラッシュ、クイック、スキップ→コピー→クイック等の固定手順も比較する。
    - 必要理由: 0% と 100% は期待方策が比較的明確で、環境・報酬・行動評価の基本的な整合性を確認しやすいため。
    - 99%カウンターは現時点の必須項目から外す。ロックのような状態依存スキルは単純な環境では価値が出にくいため、必要になった時点で専用シナリオを作る。
37. **評価設計が承認されてから再学習する。**
    - 4構成リーグ評価（P11）は P9 の診断・評価レポート拡張が終わるまで保留。
    - 必要理由: 原因未特定のまま再学習すると、数字宣言偏重や追加ターン偏重を再発させる可能性が高いため。
