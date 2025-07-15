#!/bin/bash

VENV_DIR=$HOME/.squai_env_frontend
VENV_ACTIVATE=$VENV_DIR/bin/activate

SCRIPT_DIR=$(dirname $(realpath "$0"))

if [[ ! -d $VENV_DIR ]] || [[ ! -e $VENV_ACTIVATE ]]; then
	python3 -mvenv $VENV_DIR

	source $VENV_ACTIVATE

	pip install streamlit psutil beartype rich
fi

source $VENV_ACTIVATE

USERNAME=s3811141

bash $SCRIPT_DIR/continous_hpc/enterprise_cloud/run \
	--hpc-system-url login1.capella.hpc.tu-dresden.de \
	--local-hpc-script-dir $(pwd) \
	--hpc-script-dir /home/$USERNAME/squai \
	--jumphost-url imageseg.scads.ai \
	--jumphost-username service \
	--hpc-job-name squai_job \
	--username $USERNAME \
	--sbatch_file_name backend.sh \
	--copy \
	--local-port 8500 \

