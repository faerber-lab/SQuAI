#!/bin/bash

function nice_echo {
    echo -e "\e[1;32m-> $1\e[0m"
}

VENV_DIR=$HOME/.squai_env
VENV_ACTIVATE=$VENV_DIR/bin/activate
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Setup virtual environment if needed
if [[ ! -d $VENV_DIR ]] || [[ ! -e $VENV_ACTIVATE ]]; then
    nice_echo "Virtual Environment '$VENV_DIR' not found. Trying to create it..."
    python3 -m venv "$VENV_DIR"

    source "$VENV_ACTIVATE"

    nice_echo "Installing modules from $SCRIPT_DIR/requirements.txt"
    pip install -r "$SCRIPT_DIR/requirements.txt"
    nice_echo "Done installing modules"
fi

source "$VENV_ACTIVATE"

nice_echo "cd $SCRIPT_DIR"
cd "$SCRIPT_DIR"

nice_echo "Creating log dir"
mkdir -p logs

export USE_GPU=0

nice_echo "Launch FastAPI with uvicorn (CPU mode)"
uvicorn_port=8000 uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
