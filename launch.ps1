# ================================
# OmicsToken Unified Launcher v2
# ================================

param(
    [switch]$Production,
    [switch]$SkipBrowser
)

$ErrorActionPreference = "Stop"

Write-Host "🔬 Starting OmicsToken local stack…" -ForegroundColor Cyan

# --- Configuration ---
$VenvPath = ".\.venv\Scripts\activate.ps1"
$MemuraiExe = "C:\Program Files\Memurai\memurai.exe"
$Port = 8000

if ($Production) {
    $BackendCmd = "uvicorn app:app --host 0.0.0.0 --port $Port --workers 4"
} else {
    $BackendCmd = "uvicorn app:app --reload --port $Port"
}

$CeleryCmd = "celery -A worker.celery_app worker --loglevel=info --pool=solo"
# ---------------------

# --- Validate Environment ---
Write-Host "?? Validating environment..."

# Load .env into environment so local secrets are picked up.
$envFile = Join-Path $PSScriptRoot ".env"
if (Test-Path $envFile) {
    Write-Host "?? Loading environment from .env"
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) { return }
        $parts = $line -split "=", 2
        if ($parts.Count -eq 2) {
            $key = $parts[0].Trim()
            $value = $parts[1].Trim()
            if ($key) {
                Set-Item -Path "Env:$key" -Value $value -ErrorAction SilentlyContinue
            }
        }
    }
}

$RequiredVars = @("GOOGLE_API_KEY")
$Missing = @()

foreach ($var in $RequiredVars) {
    if (-not (Test-Path ("Env:$var"))) {
        $Missing += $var
    }
}

if ($Missing.Count -gt 0) {
    Write-Host "? Missing environment variables:" -ForegroundColor Red
    $Missing | ForEach-Object { Write-Host "   - $_" -ForegroundColor Red }
    exit 1
}

Write-Host "? Environment validated" -ForegroundColor Green

# --- Activate venv ---
if (Test-Path $VenvPath) {
    Write-Host "📦 Activating virtual environment..."
    & $VenvPath
} else {
    Write-Host "❌ Could not find venv at $VenvPath" -ForegroundColor Red
    exit 1
}

# --- Check Dependencies ---
Write-Host "📦 Checking Python dependencies..."
python -c "import fastapi, celery, redis, numpy" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Missing dependencies. Run: pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

# --- Start Memurai ---
Write-Host "🧠 Checking Memurai..."

$MemuraiService = Get-Service -Name "Memurai" -ErrorAction SilentlyContinue
if ($MemuraiService -and $MemuraiService.Status -eq "Running") {
    Write-Host "✅ Memurai service already running" -ForegroundColor Green
    $redis = $null
} elseif (Test-Path $MemuraiExe) {
    Write-Host "🧠 Launching Memurai..."
    $redis = Start-Process -FilePath $MemuraiExe -ArgumentList "--port 6379" -PassThru
    Start-Sleep -Seconds 2
} else {
    Write-Host "❌ Memurai not found. Please start it manually or install from:" -ForegroundColor Red
    Write-Host "   https://www.memurai.com/" -ForegroundColor Yellow
    exit 1
}

# --- Start Celery ---
Write-Host "⚙️ Starting Celery worker..."
$celery = Start-Process powershell -ArgumentList "-NoExit", "-Command", $CeleryCmd -PassThru

Start-Sleep -Seconds 2

# --- Start Backend ---
Write-Host "🚀 Starting FastAPI backend..."
$backend = Start-Process powershell -ArgumentList "-NoExit", "-Command", $BackendCmd -PassThru

Start-Sleep -Seconds 3

# --- Health Check ---
Write-Host "🩺 Running health check..."
try {
    $null = Invoke-RestMethod -Uri "http://localhost:$Port/health" -TimeoutSec 5
    Write-Host "✅ Backend is healthy" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Health check failed (backend may still be starting)" -ForegroundColor Yellow
}

# --- Summary ---
Write-Host ""
Write-Host "=====================================" -ForegroundColor Green
Write-Host "🎉 OmicsToken Stack Running"
if ($redis) { Write-Host "   - Memurai PID:  $($redis.Id)" }
Write-Host "   - Celery PID:  $($celery.Id)"
Write-Host "   - FastAPI PID: $($backend.Id)"
Write-Host ""
Write-Host "📍 Open: http://localhost:$Port/static/upload.html"
Write-Host "📋 Logs: logs\app.log, logs\celery.log"
Write-Host "=====================================" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop all services" -ForegroundColor Cyan

# --- Open Browser ---
if (-not $SkipBrowser) {
    Start-Sleep -Seconds 1
    Start-Process "http://localhost:$Port/static/upload.html"
}

# --- Cleanup Handler ---
$CleanupScript = {
    Write-Host ""
    Write-Host "🛑 Shutting down services..." -ForegroundColor Yellow
    
    if ($redis) { Stop-Process -Id $redis.Id -Force -ErrorAction SilentlyContinue }
    if ($celery) { Stop-Process -Id $celery.Id -Force -ErrorAction SilentlyContinue }
    if ($backend) { Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue }
    
    Write-Host "✅ Cleanup complete" -ForegroundColor Green
    exit 0
}

Register-EngineEvent PowerShell.Exiting -Action $CleanupScript

# Keep script running
if ($redis) {
    Wait-Process -Id $redis.Id,$celery.Id,$backend.Id
} else {
    Wait-Process -Id $celery.Id,$backend.Id
}