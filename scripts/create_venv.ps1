$ErrorActionPreference = "Stop"

if (Test-Path ".venv") {
    Write-Host ".venv already exists."
    exit 0
}

python -m venv .venv

Write-Host "Created virtual environment at .venv"
Write-Host "Activate with: .\\.venv\\Scripts\\Activate.ps1"

