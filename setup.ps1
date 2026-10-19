$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $root

function Find-Python([string]$version) {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        $found = & py ('-' + $version) -c 'import sys; print(sys.executable)' 2>$null
        if ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $found)) { return $found }
    }
    $candidate = Join-Path $env:LOCALAPPDATA ('Programs\Python\Python' + $version.Replace('.', '') + '\python.exe')
    if (Test-Path -LiteralPath $candidate) { return $candidate }
    throw "Python $version x64 is required. Install it from https://www.python.org/downloads/windows/ and rerun setup.bat."
}

try {
    if (-not (Test-Path '.venv\Scripts\python.exe')) {
        $py310 = Find-Python '3.10'
        & $py310 -m venv .venv
    }
    if (-not (Test-Path '.venv-rvc\Scripts\python.exe')) {
        $py312 = Find-Python '3.12'
        & $py312 -m venv .venv-rvc
    }
    $svs = (Resolve-Path '.venv\Scripts\python.exe').Path
    $rvc = (Resolve-Path '.venv-rvc\Scripts\python.exe').Path
    & $svs -m pip install 'torch==2.2.2+cu118' --index-url https://download.pytorch.org/whl/cu118
    if ($LASTEXITCODE -ne 0) { throw 'SVS PyTorch installation failed.' }
    & $svs -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'SVS dependencies failed.' }
    & $rvc -m pip install 'torch==2.7.1+cu118' 'torchaudio==2.7.1+cu118' --index-url https://download.pytorch.org/whl/cu118
    if ($LASTEXITCODE -ne 0) { throw 'RVC PyTorch installation failed.' }
    & $rvc -m pip install -r requirements-rvc.txt
    if ($LASTEXITCODE -ne 0) { throw 'RVC dependencies failed.' }
    if (-not (Test-Path 'external\DiffSingerKR-space\Checkpoint\S_200000.pt')) {
        & $svs scripts\fetch_diffsingerkr.py
        if ($LASTEXITCODE -ne 0) { throw 'DiffSingerKR download failed.' }
    }
    & (Join-Path $root 'scripts\fetch_ppaso.ps1')
    if (-not (Test-Path 'external\RVC\infer\cli.py')) {
        git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git external\RVC
        if ($LASTEXITCODE -ne 0) { throw 'RVC clone failed. Install Git and retry.' }
        git -C external\RVC checkout 81eed5e8f68b6bed1789f682fe78cdd324495afc
        if ($LASTEXITCODE -ne 0) { throw 'RVC version checkout failed.' }
    }
    if (-not (Test-Path 'external\RVC\assets\hubert_base\pytorch_model.bin')) {
        & $svs scripts\fetch_rvc_assets.py
        if ($LASTEXITCODE -ne 0) { throw 'RVC asset download failed.' }
    }
    if (-not (Test-Path 'external\ffmpeg\ffmpeg.exe')) {
        & $svs scripts\fetch_ffmpeg.py
        if ($LASTEXITCODE -ne 0) { throw 'FFmpeg download failed.' }
    }
    & $rvc scripts\download_models.py
    if ($LASTEXITCODE -ne 0) { throw 'Demucs or Whisper model download failed.' }
    & $svs scripts\check_environment.py
    if ($LASTEXITCODE -ne 0) { throw 'Environment check failed.' }
    Write-Host 'Setup complete. Run run.bat.'
} catch {
    Write-Error $_
    exit 1
}
