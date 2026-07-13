@echo off
chcp 65001 > nul
cd /d "%~dp0"
python web_game.py
if errorlevel 1 (
  echo.
  echo Pythonでの起動に失敗しました。
  pause
)
