@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Environment missing. Run setup.bat first.
  pause
  exit /b 1
)
set PYTHONUTF8=1
".venv\Scripts\python.exe" -m app
if errorlevel 1 (
  echo Application failed. Review the error above.
  pause
  exit /b 1
)
