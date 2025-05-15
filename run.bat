@echo off
echo Starting Open LLM Gateway...

REM Get the directory where the script is located
SET "SCRIPT_DIR=%~dp0"

REM Define the virtual environment path relative to the script directory
SET "VENV_PATH=%SCRIPT_DIR%.venv"

REM Activate virtual environment if it exists
if exist "%VENV_PATH%\Scripts\activate.bat" (
    echo Activating virtual environment from %VENV_PATH%...
    call "%VENV_PATH%\Scripts\activate.bat"
    echo Virtual environment activated.
) else (
    echo Virtual environment not found at %VENV_PATH%. Attempting to run with system Python.
    echo It is recommended to create and use a virtual environment.
)

REM Navigate to the script's directory to ensure correct relative paths for other files
cd /D "%SCRIPT_DIR%"

REM Install dependencies if requirements.txt exists
if exist "requirements.txt" (
    echo Installing/updating dependencies from requirements.txt...
    pip install -r requirements.txt
) else (
    echo requirements.txt not found in %SCRIPT_DIR%. Skipping dependency installation.
)

REM Run the Uvicorn server
echo Starting Uvicorn server for main:app...
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

REM Pause if you want to see the output before the window closes when not run from an existing cmd
REM pause 