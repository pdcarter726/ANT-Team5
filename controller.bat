@echo off
rem Universal Computer Controller (dashboard). Extra flags pass through, e.g.  controller.bat --eeg sim
cd /d "%~dp0"
".venv\Scripts\python.exe" -m controller %*
