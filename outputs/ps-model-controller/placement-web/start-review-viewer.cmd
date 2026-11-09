@echo off
set "VIEWER_PYTHON=D:\CodexProjects\experiments\scrap-composite\work\triposr-env\Scripts\pythonw.exe"
set "VIEWER_SCRIPT=D:\CodexProjects\experiments\scrap-composite\outputs\ps-model-controller\placement-web\review_viewer.py"
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
