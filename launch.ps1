# ================================
# OmicsToken Unified Launcher
# ================================

$ErrorActionPreference = "Stop"

Write-Host "🔬 Starting OmicsToken local stack…" -ForegroundColor Cyan

# --- Configuration ---
$VenvPath = ".\.venv\Scripts\activate.ps1"
$RedisExe = "C:\Program Files\Redis\redis-server.exe"   # adjust if needed
$BackendCmd = "uvicorn app:app --reload --port 8080"
$CeleryCmd = "celery -A worker.celery_app worker --loglevel=info"
# ---------------------

# --- Activate venv ---
if (Test-Path $VenvPath) {
    Write-Host "📦 Activating virtual environment..."
    & $VenvPath
}
else {
    Write-Host "❌ Could not find venv. Expected at $VenvPath" -ForegroundColor Red
    exit 1
}

# --- Start Redis ---
Write-Host "🧠 Launching Redis..."
$redis = Start-Process -FilePath $RedisExe -ArgumentList "--port 6379" -PassThru

Start-Sleep -Seconds 1

# --- Start Celery ---
Write-Host "⚙️ Starting Celery worker..."
$celery = Start-Process powershell -ArgumentList "-NoExit", "-Command", $CeleryCmd -PassThru

Start-Sleep -Seconds 1

# --- Start Backend ---
Write-Host "🚀 Starting FastAPI backend..."
$backend = Start-Process powershell -ArgumentList "-NoExit", "-Command", $BackendCmd -PassThru

Start-Sleep -Seconds 2

Write-Host ""
Write-Host "=====================================" -ForegroundColor Green
Write-Host "🎉 OmicsToken Stack Running"
Write-Host "   - Redis PID:   $($redis.Id)"
Write-Host "   - Celery PID:  $($celery.Id)"
Write-Host "   - FastAPI PID: $($backend.Id)"
Write-Host ""
Write-Host "Open: http://localhost:8080/static/upload.html"
Write-Host "=====================================" -ForegroundColor Green

# Keep the script alive until user closes it
Wait-Process -Id $redis.Id,$celery.Id,$backend.Id
