@echo off
title Bailing Server - Production
setlocal

if not exist ".venv" goto :no_venv

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Starting Bailing Server (Production Mode)...
python server.py
pause
goto :eof

:no_venv
echo Error: Virtual environment (.venv) not found. Please run init.bat first.
pause
exit /b 1
