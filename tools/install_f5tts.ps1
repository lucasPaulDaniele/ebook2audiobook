# Installs the F5-TTS engine in its own Python environment (python_env_f5), so its pinned
# dependencies (torch 2.8, transformers…) never touch python_env used by ebook2audiobook/XTTS.
# The engine is then driven by lib/classes/tts_engines/f5tts.py through a worker process.
#
#   powershell -ExecutionPolicy Bypass -File tools\install_f5tts.ps1
param([string]$Log = '')
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$Host.UI.RawUI.WindowTitle = 'E2A - Installation du moteur F5-TTS'
if (-not $Log) { $Log = Join-Path $root 'tmp\install_f5tts.log' }
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Log) | Out-Null
[System.IO.File]::WriteAllText($Log, "Installation F5-TTS - debut $(Get-Date -Format 'HH:mm:ss')`r`n")

function Step([string]$msg) {
    Write-Host ''
    Write-Host "==> $msg" -ForegroundColor Cyan
    [System.IO.File]::AppendAllText($Log, "$(Get-Date -Format 'HH:mm:ss') $msg`r`n")
}

function Fetch([string]$url, [string]$dest) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dest) | Out-Null
    for ($i = 1; $i -le 10; $i++) {
        & curl.exe -L --fail -C - --retry 5 --retry-delay 3 --speed-limit 50000 --speed-time 30 -o $dest $url
        if ($LASTEXITCODE -eq 0 -or $LASTEXITCODE -eq 33) { return }
        Write-Host "  reprise du telechargement (essai $i)..." -ForegroundColor Yellow
    }
    throw "telechargement impossible : $url"
}

$code = 0
try {
    $uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
    if (-not $uv) { throw 'uv introuvable : installez-le (winget install astral-sh.uv)' }
    $envDir = Join-Path $root 'python_env_f5'
    $models = Join-Path $root 'models\f5tts'
    $py = Join-Path $envDir 'Scripts\python.exe'
    Step '1/5 Environnement Python 3.11 (python_env_f5)'
    & $uv venv --python 3.11 --allow-existing $envDir
    if ($LASTEXITCODE) { throw 'creation de l environnement impossible' }
    Step '2/5 PyTorch 2.8 CUDA 12.6 (~2.5 Go)'
    & $uv pip install --python $py torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
    if ($LASTEXITCODE) { throw 'installation de PyTorch impossible' }
    Step '3/5 F5-TTS 1.1.22'
    & $uv pip install --python $py f5-tts==1.1.22 torchcodec==0.7.0 soundfile
    if ($LASTEXITCODE) { throw 'installation de F5-TTS impossible' }
    Step '4/5 Modele francais RASPIAUDIO F5-French-MixedSpeakers-reduced (1.35 Go) et vocodeur vocos'
    $hf = 'https://huggingface.co'
    Fetch "$hf/RASPIAUDIO/F5-French-MixedSpeakers-reduced/resolve/main/model_last_reduced.pt" (Join-Path $models 'fr\model_last_reduced.pt')
    Fetch "$hf/RASPIAUDIO/F5-French-MixedSpeakers-reduced/resolve/main/vocab.txt" (Join-Path $models 'fr\vocab.txt')
    Fetch "$hf/charactr/vocos-mel-24khz/resolve/main/config.yaml" (Join-Path $models 'vocos\config.yaml')
    Fetch "$hf/charactr/vocos-mel-24khz/resolve/main/pytorch_model.bin" (Join-Path $models 'vocos\pytorch_model.bin')
    Step '5/5 Verification'
    & $py -c "import torch, f5_tts.api; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_arch_list())"
    if ($LASTEXITCODE) { throw "verification en echec (code $LASTEXITCODE)" }
    Step 'Installation de F5-TTS terminee'
} catch {
    $code = 1
    Write-Host "ECHEC : $_" -ForegroundColor Red
    [System.IO.File]::AppendAllText($Log, "ECHEC : $_`r`n")
}
[System.IO.File]::AppendAllText($Log, "=== TERMINE (code $code) a $(Get-Date -Format 'HH:mm:ss') ===`r`n")
if ($code) { Start-Sleep -Seconds 10 } else { Start-Sleep -Seconds 3 }
exit $code
