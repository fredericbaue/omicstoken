@echo off
:: ========================================================
:: vss.bat - OMICSTOKEN Session Bootstrapper v1.2
:: Intent: Repo Sync -> Env Setup -> Context Packing -> Launch
:: Philosophy: Fail fast, boring solutions, zero friction.
:: ========================================================

:: 1. CONFIGURATION
set "REPO_PATH=C:\Users\test\metabo-mvp"
set "REPO_URL=https://github.com/fredericbaue/omicstoken"
set "TARGET_BRANCH=main"
set "CONTEXT_OUTPUT=_SESSION_CONTEXT.txt"

echo [VSS] Initializing OMICSTOKEN Session...
echo [VSS] Target: %REPO_PATH%

:: 2. REPO VALIDATION
if not exist "%REPO_PATH%" (
    echo [ERROR] Repository not found at %REPO_PATH%.
    pause
    exit /b 1
)

cd /d "%REPO_PATH%"

:: 3. GIT SYNCHRONIZATION
echo [GIT] Fetching origin...
git fetch origin >nul 2>&1
echo [GIT] Pulling %TARGET_BRANCH%...
git pull origin %TARGET_BRANCH%
if %errorlevel% neq 0 (
    echo [ERROR] Git pull failed. Resolve conflicts.
    pause
    exit /b 1
)

:: 4. RUNTIME SETUP
if not exist .venv (
    echo [ENV] Creating .venv...
    python -m venv .venv
)

echo [ENV] Activating .venv...
call .venv\Scripts\activate

echo [ENV] Syncing dependencies...
pip install -r requirements.txt >nul 2>&1

:: 4.5 PACK CONTEXT (The Brain)
echo [CTX] Packing context files...
if exist pack_context.py (
    python pack_context.py
) else (
    echo [WARN] pack_context.py not found. Skipping auto-pack.
)

:: 5. LAUNCH
echo [VSS] Opening VS Code...
:: Opens the folder + the packed context file for immediate copy-paste usage
code . %CONTEXT_OUTPUT%

echo [VSS] Session Ready.
exit /b 0