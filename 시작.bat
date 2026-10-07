@echo off
setlocal
cd /d "%~dp0"
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not defined PY if exist "C:\Users\etenenmz1\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" set "PY=C:\Users\etenenmz1\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if not defined PY (
  echo Python 3 is required. Install Python from python.org and run this launcher again.
  pause
  exit /b 1
)
%PY% server.py --open-browser
