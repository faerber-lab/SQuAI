#!/bin/bash

#SBATCH --job-name=fastapi-backend
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=8:00:00
#SBATCH --output=logs/fastapi_%j.log

function nice_echo {
	echo -e "\e[1;32m-> $1\e[0m"
}

VENV_DIR=$HOME/.squai_env
VENV_ACTIVATE=$VENV_DIR/bin/activate

SCRIPT_DIR=$(dirname $(realpath "$0"))

if [[ ! -z $SLURM_JOB_ID ]]; then
	SCRIPT_DIR=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}' | sed -e 's#backend.sh##')

	nice_echo "-> Loading modules"
	module load release/24.04 GCC/12.3.0 OpenMPI/4.1.5 PyTorch/2.1.2
	nice_echo "-> Done loading modules"
fi

if [[ ! -d $VENV_DIR ]] || [[ ! -e $VENV_ACTIVATE ]]; then
	nice_echo "-> Virtual Environment '$VENV_DIR' not found. Trying to create it..."
	python3 -mvenv $VENV_DIR

	source $VENV_ACTIVATE

	nice_echo "-> Installing modules from $SCRIPT_DIR/requirements.txt"
	pip install -r $SCRIPT_DIR/requirements.txt
	nice_echo "-> Done installing modules"
fi

source $VENV_ACTIVATE

nice_echo "cd $SCRIPT_DIR"
cd $SCRIPT_DIR

nice_echo "Creating log dir $SCRIPT_DIR"
mkdir -p logs

nice_echo "Launch FastAPI with uvicorn"
uvicorn_port=8000 uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1

nice_echo "Launch main server on port 8501, broadcasting to 0.0.0.0"
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
