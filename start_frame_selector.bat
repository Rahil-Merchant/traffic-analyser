@echo off
setlocal
title Frame Selector
color 0B

:: Same environment as start_server.bat
set CONDA_ENV=traffic_env_py310
set PORT=8090
set PROJECT_DIR=%~dp0

echo.
echo =========================================
echo    Frame Selector
echo    picks the frames worth labelling
echo =========================================
echo.

where conda >nul 2>&1
if errorlevel 1 (
    echo ERROR: Conda is not found in PATH
    pause
    exit /b 1
)

call conda activate %CONDA_ENV%
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment '%CONDA_ENV%'
    pause
    exit /b 1
)

cd /d "%PROJECT_DIR%"
echo Opening http://127.0.0.1:%PORT% in your browser.
echo Keep this window open while you use the tool; close it to stop.
echo.
python -m frame_selector gui --port %PORT%

echo.
echo Frame Selector stopped.
pause
