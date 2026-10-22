@echo off
title Push Self-Driving Car AI to GitHub
cd /d "%~dp0"
echo =======================================================
echo   Pushing Self-Driving Car AI to GitHub
echo =======================================================
echo.
git push -u origin main
if %ERRORLEVEL% EQU 0 (
    echo.
    echo =======================================================
    echo   SUCCESSFULLY PUSHED TO GITHUB!
    echo   Repo: https://github.com/jayesh456654/self-driving-car-ai
    echo =======================================================
) else (
    echo.
    echo Make sure you clicked 'Create repository' on GitHub first!
)
echo.
pause
