#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define the virtual environment path relative to the script directory
VENV_PATH="$SCRIPT_DIR/.venv"

# Activate virtual environment if it exists
if [ -d "$VENV_PATH" ]; then
  echo "Activating virtual environment from $VENV_PATH..."
  source "$VENV_PATH/bin/activate"
  echo "Virtual environment activated."
else
  echo "Virtual environment not found at $VENV_PATH. Attempting to run with system Python."
  echo "It is recommended to create and use a virtual environment."
fi

# Navigate to the script's directory to ensure correct relative paths for other files
cd "$SCRIPT_DIR"

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
  echo "Installing/updating dependencies from requirements.txt..."
  pip install -r requirements.txt
else
  echo "requirements.txt not found in $SCRIPT_DIR. Skipping dependency installation."
fi

# Run the Uvicorn server
echo "Starting Uvicorn server for main:app..."
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Deactivate virtual environment upon exiting (optional)
# if [ -d "$VENV_PATH" ]; then
#   deactivate
#   echo "Virtual environment deactivated."
# fi 