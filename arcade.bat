@echo off
rem Hands-free arcade: serves games/ on http://127.0.0.1:8765 and opens the browser. Extra flags pass through.
.venv\Scripts\python.exe games\serve.py %*
