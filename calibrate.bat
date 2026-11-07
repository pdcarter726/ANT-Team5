@echo off
rem 2-minute calibration. Saves the new profile as "calibrated" (the base profile is untouched). Needs the live EEG stream.
"%~dp0.venv\Scripts\python.exe" "%~dp0src\calibrate.py" --base default --name calibrated %*
