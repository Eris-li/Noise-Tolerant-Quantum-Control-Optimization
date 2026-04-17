$ErrorActionPreference = "Stop"

$env:MPLCONFIGDIR = Join-Path $PSScriptRoot "..\\.cache\\matplotlib"
$venvPython = Join-Path $PSScriptRoot "..\\.venv.win\\Scripts\\python.exe"

if (-not (Test-Path $env:MPLCONFIGDIR)) {
    New-Item -ItemType Directory -Path $env:MPLCONFIGDIR -Force | Out-Null
}

& $venvPython @args
