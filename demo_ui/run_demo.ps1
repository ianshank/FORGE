<#
.SYNOPSIS
    One-click launcher for the FORGE Demo UI.

.DESCRIPTION
    1. Installs Python dependencies
    2. Starts the FastAPI server on http://127.0.0.1:8765
    3. Opens a browser to the demo UI

.USAGE
    .\demo_ui\run_demo.ps1

    Optional flags:
      -Port 9000         # Use a different port (default: 8765)
      -NoBrowser        # Don't auto-open the browser
      -SkipInstall      # Skip pip install step
#>

param(
    [int]$Port = 8765,
    [switch]$NoBrowser,
    [switch]$SkipInstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$FORGE_ROOT = Split-Path -Parent $PSScriptRoot
$REQUIREMENTS = Join-Path $PSScriptRoot "backend\requirements.txt"
$URL = "http://127.0.0.1:$Port"

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "   FORGE Demo UI" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# ----- Step 1: Install dependencies -----
if (-not $SkipInstall) {
    Write-Host "[1/3] Installing dependencies..." -ForegroundColor Yellow
    pip install -r $REQUIREMENTS --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Error "pip install failed. Check your Python environment."
        exit 1
    }
    Write-Host "      Done." -ForegroundColor Green
} else {
    Write-Host "[1/3] Skipping dependency install (-SkipInstall)." -ForegroundColor DarkGray
}

# ----- Step 2: Launch server -----
Write-Host "[2/3] Starting FastAPI server on $URL ..." -ForegroundColor Yellow

$serverProc = Start-Process -FilePath python -ArgumentList `
    "-m", "uvicorn",
    "demo_ui.backend.main:app",
    "--host", "127.0.0.1",
    "--port", "$Port",
    "--reload",
    "--log-level", "info" `
    -WorkingDirectory $FORGE_ROOT `
    -PassThru -NoNewWindow

# Give the server time to bind
Start-Sleep -Seconds 2

if ($serverProc.HasExited) {
    Write-Error "Server process exited prematurely (code $($serverProc.ExitCode))."
    exit 1
}

Write-Host "      Server PID: $($serverProc.Id)" -ForegroundColor Green

# ----- Step 3: Open browser -----
if (-not $NoBrowser) {
    Write-Host "[3/3] Opening browser at $URL ..." -ForegroundColor Yellow
    Start-Process $URL
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "   FORGE Demo UI is running at: $URL" -ForegroundColor White
Write-Host "   Press Ctrl+C to stop the server." -ForegroundColor DarkGray
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Wait until user presses Ctrl+C
try {
    while ($true) { Start-Sleep -Seconds 5 }
} finally {
    Write-Host "`nShutting down server (PID $($serverProc.Id))..." -ForegroundColor Yellow
    Stop-Process -Id $serverProc.Id -Force -ErrorAction SilentlyContinue
    Write-Host "Done. Goodbye!" -ForegroundColor Green
}
