#!/bin/bash
#SBATCH --job-name=fastapi-backend
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=8:00:00
#SBATCH --output=logs/fastapi_%j.log

SCRIPT_DIR=$(dirname $(realpath "$0"))

if [[ ! -z $SLURM_JOB_ID ]]; then
	SCRIPT_DIR=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')

	module load release/24.04 GCC/12.3.0 OpenMPI/4.1.5 PyTorch/2.1.2

	VENV_DIR=$HOME/.squai_env
	VENV_ACTIVATE=$VENV_DIR/bin/activate
fi

if [[ ! -d $VENV_DIR ]] || [[ ! -e $VENV_ACTIVATE ]]; then
	python3 -mvenv $VENV_DIR
fi

source $VENV_ACTIVATE

# Create log directory if it doesn't exist
mkdir -p $SCRIPT_DIR/logs

# Launch FastAPI with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1

streamlit app.py --server.port 8501 --server.address 0.0.0.0
