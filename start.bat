@echo off
SETLOCAL
call .\.venv\Scripts\Activate.bat
start "Celery Worker" /min celery -A worker.celery_app worker --loglevel=info --pool=solo
start "FastAPI" /min uvicorn app:app --host 127.0.0.1 --port 8080 --reload
ENDLOCAL
