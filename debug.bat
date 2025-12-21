@echo off
title Bailing Server - Debug
setlocal

if not exist ".venv" goto :no_venv

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Starting Bailing Server (Debug Mode with Reload)...
python server.py --debug
pause
goto :eof

:no_venv
echo Error: Virtual environment (.venv) not found. Please run init.bat first.
pause
exit /b 1
