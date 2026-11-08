@echo off
rem Controller with simulated EEG (recorded trials injected live from the side panel). Extra flags pass through.
"%~dp0.venv\Scripts\python.exe" -m controller --eeg inject %*
