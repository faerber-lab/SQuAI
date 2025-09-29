#!/bin/bash

# Defaults
VENV_DIR=$HOME/.squai_env_frontend
VENV_ACTIVATE=$VENV_DIR/bin/activate
SCRIPT_DIR=$(dirname "$(realpath "$0")")

USERNAME="squai"
CLUSTER="capella"
FALLBACK_CLUSTER="alpha"
HPC_URL=""
LOCAL_PORT="8000"

# Show help
function show_help {
	echo "Usage: $(basename "$0") [options]"
	echo ""
	echo "Options:"
	echo "  --username USERNAME           HPC username (default: $USERNAME)"
	echo "  --cluster CLUSTER             Cluster name (e.g. capella, alpha) (default: $CLUSTER)"
	echo "  --hpc-url URL                 Full HPC login URL (overrides --cluster)"
	echo "  --local-port PORT             Local port for forwarding (default: $LOCAL_PORT)"
	echo "  --help                        Show this help message and exit"
	exit 0
}

# Argument parsing
while [[ $# -gt 0 ]]; do
	case $1 in
		--username)
			USERNAME="$2"
			shift 2
			;;
		--cluster)
			CLUSTER="$2"
			shift 2
			;;
		--hpc-url)
			HPC_URL="$2"
			shift 2
			;;
		--local-port)
			LOCAL_PORT="$2"
			shift 2
			;;
		--help)
			show_help
			;;
		*)
			echo "Unknown option: $1"
			echo "Use --help for usage information."
			exit 1
			;;
	esac
done

# Derive URL if not given
if [[ -z "$HPC_URL" ]]; then
	HPC_URL="login1.${CLUSTER}.hpc.tu-dresden.de"
fi
	
    FALLBACK_HPC_URL="login1.${FALLBACK_CLUSTER}.hpc.tu-dresden.de"

# Setup virtual environment if needed
if [[ ! -d $VENV_DIR ]] || [[ ! -e $VENV_ACTIVATE ]]; then
	python3 -m venv "$VENV_DIR"
	source "$VENV_ACTIVATE"
	pip install streamlit psutil beartype rich
fi

source "$VENV_ACTIVATE"

# Run backend script
bash "$SCRIPT_DIR/continous_hpc/enterprise_cloud/run" \
	--hpc-system-url "$HPC_URL" \
	--local-hpc-script-dir "$(pwd)" \
	--hpc-script-dir "/home/$USERNAME/squai" \
	--hpc-job-name "squai_fastapi_backend" \
	--username "$USERNAME" \
	--sbatch_file_name "backend.sh" \
    --fallback-system-url "$FALLBACK_HPC_URL" \
	--copy \
	--local-port "$LOCAL_PORT"
