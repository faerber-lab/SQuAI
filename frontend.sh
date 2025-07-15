#!/bin/bash

VENV_DIR=$HOME/.squai_env_frontend
VENV_ACTIVATE=$VENV_DIR/bin/activate

SCRIPT_DIR=$(dirname $(realpath "$0"))

if [[ ! -d $VENV_DIR ]] || [[ ! -e $VENV_ACTIVATE ]]; then
	python3 -mvenv $VENV_DIR

	source $VENV_ACTIVATE

	pip install streamlit
fi

source $VENV_ACTIVATE

streamlit run $SCRIPT_DIR/app.py --server.port 8501 --server.address 0.0.0.0
