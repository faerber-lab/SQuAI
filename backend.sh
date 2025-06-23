#!/bin/bash
#SBATCH --job-name=fastapi-backend
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=8:00:00
#SBATCH --output=logs/fastapi_%j.log


# Load modules and activate environment
module load release/24.04 GCC/12.3.0 OpenMPI/4.1.5
module load PyTorch/2.1.2

source /home/inbe405h/env/bin/activate

# Create log directory if it doesn't exist
mkdir -p logs

# Launch FastAPI with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1

streamlit app.py --server.port 8501 --server.address 0.0.0.0
