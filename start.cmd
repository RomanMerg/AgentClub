@echo off
rem Starts the Agent Club server on http://localhost:8765 and opens it.
cd /d "%~dp0"
start "" cmd /c "timeout /t 2 >nul & start http://localhost:8765/"
python -m club --port 8765
