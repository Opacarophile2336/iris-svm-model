@echo off
title HMNC_PRO Frontend (Port 2336)
echo ======================================================
echo   HMNC_PRO - Starting Frontend on http://localhost:2336
echo ======================================================
cd /d "%~dp0frontend"
npm run dev
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Frontend encountered an error.
    pause
)
