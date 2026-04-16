$ErrorActionPreference = "Stop"

$env:MPLCONFIGDIR = Join-Path $PSScriptRoot "..\\.cache\\matplotlib"

if (-not (Test-Path $env:MPLCONFIGDIR)) {
    New-Item -ItemType Directory -Path $env:MPLCONFIGDIR -Force | Out-Null
}

& ".\.venv\Scripts\python.exe" @args
