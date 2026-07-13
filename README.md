# ユビスマ Complete — 対戦AI & ブラウザゲーム

指スマ(ユビスマ)の **Complete ルール**（ミラー・リバーシ OFF）を対象にした、強い対戦AIと、それと遊べるブラウザゲームです。
AIは「深さ制限のLP均衡バックアップ探索 ＋ 価値ネットワーク（自己対戦で学習）」で手を選びます。

---

## 🎮 すぐに遊ぶ（Windows・インストール不要）

1. **[最新リリースをダウンロード](https://github.com/ymtmyst/Yubisuma_Complete/releases/latest)** して zip を解凍
2. フォルダ内の **`ユビスマ対戦.exe`** をダブルクリック
3. 自動でブラウザに対戦画面が開きます（開かない場合は表示された `http://127.0.0.1:8000` を開く）
4. 終了は、起動した黒いコンソール画面で `Ctrl+C`

> Python も GPU も不要。初回起動時、AIの読み込みに10秒ほどかかります。

### あそびかた
- 難易度は **やさしい / ふつう / つよい** の3段階（AIの手加減＝ランダム率）
- **こども / おとな** 表示切替、スキルの説明ポップアップ、ビジュアルなルールブック内蔵
- 形勢メーター・戦績記録・効果音つき
- 自分のターンは数字/スキルと親指の本数を宣言。相手ターンは相手の宣言が見えないままリアクションを選びます（同時手番）

---

## 🧑‍💻 ソースから動かす（開発者向け）

```bash
# 依存（遊ぶだけなら軽量版でOK）
pip install -r requirements-play.txt

# 対戦サーバー起動 → ブラウザで http://127.0.0.1:8000
python -m complete_ai.play_server

# ターミナルで対戦（CLI）
python -m complete_ai.play_cli
```

配布用exeのビルド手順は [`EXE化とGitHub公開の手順.md`](EXE化とGitHub公開の手順.md) を参照。

---

## 🧠 AIのしくみ（概要）

- ゲームは **完全情報・決定論・同時手番・ゼロ和** のマルコフゲーム（乱数・隠れ情報なし）
- 全状態の列挙は非現実的（10^7 以上）なため、**深さ制限探索の葉を価値ネットで評価**し、同時手番の各ノードは **LP（線形計画）でナッシュ均衡値にバックアップ**（AlphaZeroの max を LP に置換）
- 学習は **fitted Nash-VI 自己対戦**。着手はルートの混合戦略からサンプリング（決定論プレイの病理を回避）
- 非搾取性を最良反応（BR）攻撃者で実測検証済み

### 主なコード
| 場所 | 役割 |
|---|---|
| `complete_solver/` | ルールエンジン（Numbaでビットパック高速化した `packed_engine` 含む） |
| `complete_ai/batched_search.py` | 価値ネット葉＋LPバックアップ探索 |
| `complete_ai/generation_loop.py` | 自己対戦の世代学習ループ |
| `complete_ai/play_server.py` ＋ `complete_ai/webplay/` | ブラウザ対戦（stdlibサーバー＋静的UI） |
| `models/value_latest.pt` | 学習済み価値ネット |

---

## 📚 ドキュメント
- [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) — プロジェクト全体像
- [`AI_MASTER_PLAN_V2.md`](AI_MASTER_PLAN_V2.md) — 探索中心アーキテクチャの設計
- [`WORK_LOG.md`](WORK_LOG.md) — 開発の詳細な作業ログ
- [`FUTURE_IDEAS.md`](FUTURE_IDEAS.md) — 今後のアイデア

---

*対象ルール: ユビスマ Complete（ミラー・リバーシ OFF 固定）*
