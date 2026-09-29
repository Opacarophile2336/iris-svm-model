@echo off
title HMNC_PRO Backend (Port 2006)
echo ======================================================
echo   HMNC_PRO - Starting Backend on http://127.0.0.1:2006
echo ======================================================
cd /d "%~dp0backend"
python -m uvicorn main:app --host 127.0.0.1 --port 2006 --reload
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Backend encountered an error.
    pause
)
