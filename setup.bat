@echo off
setlocal
where pwsh >nul 2>&1
if %errorlevel%==0 (
  pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
) else (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
)
if errorlevel 1 (
  echo Setup failed. See the error above.
  pause
  exit /b 1
)
echo Setup complete.
pause
