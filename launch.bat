@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [setup] Creating virtual environment...
    py -m venv .venv || goto :python_error
)

echo [setup] Upgrading pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :install_error

echo [setup] Installing requirements...
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :install_error

echo [run] Launching gaze media controller...
".venv\Scripts\python.exe" gaze_media_controller.py
goto :end

:python_error
echo.
echo Python launcher (py) not found.
echo Install Python 3.10+ from https://www.python.org/downloads/
goto :pause

:install_error
echo.
echo Failed to install dependencies.
goto :pause

:pause
pause

:end
endlocal
