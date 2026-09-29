@echo off
title IrisAI Studio - One-Port Production (Port 2336)
echo ======================================================
echo   IrisAI Studio - Starting One-Port Production Server
echo   Port: 2336 (SPA UI + API + Docs)
echo ======================================================
cd /d "%~dp0"
python run_production.py %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Production server encountered an error or stopped.
    pause
)
