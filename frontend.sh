#!/bin/bash

function nice_echo {
	echo -e "\e[1;32m-> $1\e[0m"
}

VENV_DIR=$HOME/.squai_env_frontend
VENV_ACTIVATE=$VENV_DIR/bin/activate

SCRIPT_DIR=$(dirname $(realpath "$0"))

if [[ ! -d $VENV_DIR ]] || [[ ! -e $VENV_ACTIVATE ]]; then
	nice_echo "Virtual Environment '$VENV_DIR' not found. Trying to create it..."
	python3 -mvenv $VENV_DIR

	source $VENV_ACTIVATE

	nice_echo "Installing modules"
	pip install streamlit
	nice_echo "Done installing modules"
fi

source $VENV_ACTIVATE

nice_echo "Starting backend server as a background job"
bash $SCRIPT_DIR/start_backend_from_enterprise_cloud.sh &

nice_echo "Waiting for localhost:8501..."
until nc -z localhost 8501; do
	sleep 1
done
nice_echo "localhost:8501 is reachable!"

nice_echo "Starting streamlit script $SCRIPT_DIR/app.py"

streamlit run $SCRIPT_DIR/app.py --server.port 8501 --server.address 0.0.0.0
