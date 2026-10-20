$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $root '.venv\Scripts\python.exe'
$optional = Join-Path $root '.venv-basicpitch'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run setup.bat first.' }
if (-not (Test-Path -LiteralPath (Join-Path $optional 'Scripts\python.exe'))) {
    & $python -m venv $optional
    if ($LASTEXITCODE -ne 0) { throw 'Basic Pitch environment creation failed.' }
}
& (Join-Path $optional 'Scripts\python.exe') -m pip install 'basic-pitch[onnx]==0.4.0'
if ($LASTEXITCODE -ne 0) { throw 'Basic Pitch installation failed.' }
Write-Host 'Basic Pitch installed. Generate a MIDI draft from the Source tab.'
