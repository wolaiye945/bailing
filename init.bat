@echo off
setlocal enabledelayedexpansion

echo ==================================================
echo   Bailing Project Initialization
echo ==================================================

echo [1/5] Checking environment...
python --version >nul 2>&1
if errorlevel 1 goto :no_python

ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo Warning: ffmpeg is not installed or not in PATH. 
    echo Some features (ASR/TTS) may not work correctly.
)

echo [2/5] Creating virtual environment (.venv)...
if exist ".venv" (
    echo .venv already exists.
) else (
    python -m venv .venv
    echo .venv created.
)

echo [3/5] Activating virtual environment and installing dependencies...
if not exist ".venv\Scripts\activate.bat" goto :venv_error

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if exist "requirements.txt" (
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt
) else (
    echo Warning: requirements.txt not found. Skipping dependency installation.
)

echo [4/5] Preparing configuration...
if exist "config\config.yaml" (
    echo config\config.yaml already exists.
) else (
    if exist "config\config.yaml.template" (
        copy "config\config.yaml.template" "config\config.yaml"
        echo config\config.yaml created from template.
    ) else (
        echo Warning: config\config.yaml.template not found.
    )
)

echo [5/5] Creating temporary directory...
if not exist "tmp" mkdir "tmp"
if exist "tmp" echo tmp directory ready.

echo ==================================================
echo   Initialization complete!
echo   You can now run the project using run.bat or debug.bat
echo ==================================================
pause
goto :eof

:no_python
echo Error: Python is not installed or not in PATH.
pause
exit /b 1

:venv_error
echo Error: Virtual environment was not created correctly.
pause
exit /b 1
