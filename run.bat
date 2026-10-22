@echo off
title Self-Driving Car AI Simulation
cd /d "%~dp0"
echo ============================================
echo   Self-Driving Car AI Simulation
echo   Neural Networks + Genetic Algorithms
echo ============================================
echo.
echo Starting simulation...
echo.
python main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Something went wrong. Press any key to exit.
    pause >nul
)
