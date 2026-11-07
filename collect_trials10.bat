@echo off
REM Short session 1: 10 trials per class (about 10 min). Extra flags pass through, e.g. --cue-only --subject S01
cd /d "%~dp0"
.venv\Scripts\python.exe src\collect.py --session 1 --trials 10 %*
