@echo off
set "VIEWER_PYTHON=%~dp0..\..\..\work\triposr-env\Scripts\pythonw.exe"
set "VIEWER_SCRIPT=%~dp0review_viewer.py"
if not exist "%VIEWER_PYTHON%" (
  echo The local Python environment is missing.
  pause
  exit /b 1
)
if not exist "%VIEWER_SCRIPT%" (
  echo The viewer files are missing.
  pause
  exit /b 1
)
start "" "%VIEWER_PYTHON%" "%VIEWER_SCRIPT%" --open
