# EXE化して GitHub で配る手順

他の人が **Python なしで** ダブルクリックだけで遊べる形（exe）を作り、
GitHub Releases で配布するための手順です。

## 1. 準備（ビルドする自分のPCで1回）

```powershell
cd Complete
pip install pyinstaller
# （配布サイズを小さくしたい場合は CPU版 torch を入れておく）
# pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 2. ビルド

```powershell
cd Complete
pyinstaller yubisuma_game.spec
```

- 出来上がり: `dist/ユビスマ対戦/` フォルダ（この中に `ユビスマ対戦.exe` と必要ファイル一式）
- 中の `ユビスマ対戦.exe` をダブルクリック → ブラウザで対戦画面が開けば成功

## 3. 動作テスト（重要）

- **Python を入れていない別PC**（または別ユーザー）で `dist/ユビスマ対戦/` を丸ごとコピーして起動テスト。
- torch / numba を含むため、初回起動は十数秒かかります。コンソール（黒い画面）に
  `準備完了。` と URL が出て、ブラウザが開けばOK。

## 4. GitHub で配る

1. `dist/ユビスマ対戦/` フォルダを **zip 圧縮**（例: `ユビスマ対戦_win.zip`）。
2. GitHub リポジトリの **Releases → Draft a new release** で、その zip を添付。
3. 説明に「解凍して `ユビスマ対戦.exe` をダブルクリック」と書く。

> ソースコードも上げる場合は、`models/value_latest.pt`（AIの頭脳・約2MB）を
> 必ず含めてください。これが無いとAIが動きません。
> `archive_rules_v*/` や `results/_legacy/` は配布に不要なので `.gitignore` 済みでもOK。

## 補足・注意

- **サイズ**: torch を含むため展開後 数百MB〜1GB程度になります。CPU版 torch を使うと小さくなります。
- **onefile より onedir**: 単一exe（onefile）は torch/numba で不安定・起動が遅いため、
  このspecは onedir（フォルダ配布）にしています。
- ビルドが `numba` や `scipy` で失敗する場合は、`yubisuma_game.spec` の `hiddenimports` に
  足りないモジュールを追記してください。
- Mac/Linux 版が要る場合は、その OS 上で同じ手順を実行すればそれぞれの実行ファイルが作れます。
