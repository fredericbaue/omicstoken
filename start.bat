@echo off
SETLOCAL
call .\.venv\Scripts\Activate.bat
python -m uvicorn app:app --host 127.0.0.1 --port 8080 --reload
ENDLOCAL
