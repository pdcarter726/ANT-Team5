@echo off
REM Full session 1: 25 trials per class (about 24 min). Extra flags pass through, e.g. --cue-only --subject S01
cd /d "%~dp0"
.venv\Scripts\python.exe src\collect.py --session 1 --trials 25 %*
