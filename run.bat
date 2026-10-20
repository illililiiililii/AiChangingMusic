@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Environment missing. Run setup.bat first.
  pause
  exit /b 1
)
set PYTHONUTF8=1
if not exist "logs" mkdir "logs"
echo [%date% %time%] Application started>>"logs\console.log"
".venv\Scripts\python.exe" -m app >>"logs\console.log" 2>&1
set "APP_EXIT=%errorlevel%"
echo [%date% %time%] Application exited with code %APP_EXIT%>>"logs\console.log"
if not "%APP_EXIT%"=="0" (
  echo Application failed. Review logs\console.log, logs\crash-*.log and logs\app.log.
  pause
  exit /b %APP_EXIT%
)
