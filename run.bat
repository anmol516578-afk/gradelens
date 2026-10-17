@echo off
rem GradeLens one-command start (Windows):
rem   creates a virtual environment, installs dependencies, generates sample data if needed, opens the app.
rem Usage: double-click run.bat, or run it from a terminal.  (set PORT=8600 first to use another port)
setlocal EnableExtensions
cd /d "%~dp0"

set "PY="
for %%V in (3.11 3.12 3.13) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo GradeLens needs Python 3.11 or newer.
  echo Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating a virtual environment in .venv ...
  %PY% -m venv .venv || goto :fail
)

rem Reinstall only when requirements.txt has changed since the last successful install.
fc /b requirements.txt ".venv\requirements.installed" >nul 2>&1
if errorlevel 1 (
  echo Installing dependencies ^(the first run takes a few minutes^) ...
  ".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
  copy /y requirements.txt ".venv\requirements.installed" >nul
)

if not exist "sample_data\marks_long.csv" (
  echo Generating sample data ...
  ".venv\Scripts\python.exe" scripts\generate_sample_data.py >nul || goto :fail
)

if exist ".env" echo Using settings from .env
if "%PORT%"=="" set "PORT=8501"
echo Starting GradeLens at http://localhost:%PORT%  (press Ctrl+C to stop)
".venv\Scripts\python.exe" -m streamlit run app.py --server.port %PORT% %*
exit /b %errorlevel%

:fail
echo.
echo Setup failed - see the messages above.
pause
exit /b 1
