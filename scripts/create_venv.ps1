$ErrorActionPreference = "Stop"

$venvPath = ".venv.win"

if (Test-Path $venvPath) {
    Write-Host "$venvPath already exists."
    exit 0
}

python -m venv $venvPath

Write-Host "Created virtual environment at $venvPath"
Write-Host "Activate with: .\\$venvPath\\Scripts\\Activate.ps1"
