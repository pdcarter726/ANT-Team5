@echo off
rem Live EEG, nothing is sent to the OS (log only). Extra flags pass through.
"%~dp0.venv\Scripts\python.exe" "%~dp0src\realtime.py" --profile default --dry-run %*
