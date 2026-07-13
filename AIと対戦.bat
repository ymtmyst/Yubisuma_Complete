@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo Starting Yubisuma AI battle... (loading the AI takes 10-20 seconds; the browser opens automatically)
python -m complete_ai.play_server
if errorlevel 1 (
  echo.
  echo Failed to start. Please check that Python is installed and on PATH.
  pause
)
