@echo off
setlocal enabledelayedexpansion
title EMAAN Urban Reconnaissance
color 0B

:: Configuration
set CONDA_ENV=traffic_env_py310
set BACKEND_PORT=8080
set PROJECT_DIR=%~dp0

echo.
echo =========================================
echo    EMAAN Urban Reconnaissance
echo    Advanced Traffic Analysis Platform
echo =========================================
echo.

:: Check if conda is available
where conda >nul 2>&1
if errorlevel 1 (
    echo ERROR: Conda is not found in PATH
    echo Please install Anaconda/Miniconda or add conda to PATH
    echo.
    pause
    exit /b 1
)

:: Check if environment exists
echo [INIT] Checking environment...
conda info --envs | findstr "%CONDA_ENV%" >nul
if errorlevel 1 (
    echo ERROR: Conda environment '%CONDA_ENV%' not found
    echo Available environments:
    conda info --envs
    echo.
    pause
    exit /b 1
)

:: Activate environment
echo [1/4] Activating Python environment '%CONDA_ENV%'...
call conda activate %CONDA_ENV%
if errorlevel 1 (
    echo ERROR: Failed to activate environment
    pause
    exit /b 1
)
echo > Environment activated successfully

:: Check if backend files exist
echo [2/4] Checking backend files...
if not exist "%PROJECT_DIR%dev_server.py" (
    echo ERROR: Backend file 'dev_server.py' not found
    echo Current directory: %PROJECT_DIR%
    echo.
    pause
    exit /b 1
)
echo > Backend files found

:: Start backend server
echo [3/4] Starting backend server...
cd /d "%PROJECT_DIR%"
start "EMAAN-Backend" /min cmd /k "title EMAAN Backend Server && conda activate %CONDA_ENV% && echo Starting backend on port %BACKEND_PORT%... && python dev_server.py"

echo > Backend server initiated
echo   > Running on http://localhost:%BACKEND_PORT%
echo   > Check backend window if issues occur

:: Wait for backend to start
echo [4/4] Starting frontend application...
echo   > Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

:: Check if frontend exists
if not exist "%PROJECT_DIR%frontend\EMAAN Urban Reconnaissance.exe" (
    echo WARNING: Frontend executable not found
    echo Expected: %PROJECT_DIR%frontend\EMAAN Urban Reconnaissance.exe
    echo.
    echo Opening in web browser instead...
    timeout /t 2 /nobreak >nul
    start "" "http://localhost:%BACKEND_PORT%"
    echo.
    echo Press any key when done...
    pause >nul
) else (
    echo > Starting desktop application...
    cd "%PROJECT_DIR%frontend"
    start /wait "" "EMAAN Urban Reconnaissance.exe"
)

:: Cleanup
echo.
echo [CLEANUP] Application closed. Stopping services...
taskkill /f /fi "WINDOWTITLE eq EMAAN-Backend*" >nul 2>&1
taskkill /f /im python.exe /fi "COMMANDLINE eq *dev_server.py*" >nul 2>&1

echo.
echo =========================================
echo         Session Complete
echo    Backend services stopped
echo =========================================
echo.
echo Press any key to exit...
pause >nul