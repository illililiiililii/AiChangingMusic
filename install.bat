@echo off
call "%~dp0setup.bat"
if errorlevel 1 exit /b 1
if /I "%~1"=="--basic-pitch" (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_basic_pitch.ps1"
  if errorlevel 1 exit /b 1
)
