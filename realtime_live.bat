@echo off
rem Live EEG, REAL mouse/keyboard actions. Kill: Ctrl+Alt+Q. Extra flags pass through.
"%~dp0.venv\Scripts\python.exe" "%~dp0src\realtime.py" --profile default %*
