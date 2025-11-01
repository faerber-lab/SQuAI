#!/bin/bash

function nice_echo {
    echo -e "\e[1;32m-> $1\e[0m"
}

# Defaults
VENV_DIR=$HOME/.squai_env_frontend
VENV_ACTIVATE=$VENV_DIR/bin/activate
SCRIPT_DIR=$(dirname "$(realpath "$0")")

DEFAULT_USER=$(grep '^username' "$(cd "$(dirname "$0")" && pwd)/defaults.ini" | sed -e 's#.*=##')
DEFAULT_PARTITION=$(grep '^partition' "$(cd "$(dirname "$0")" && pwd)/defaults.ini" | sed -e 's#.*=##')

USERNAME=$DEFAULT_USER
CLUSTER=$DEFAULT_PARTITION
HPC_URL=""
LOCAL_PORT="8000"
WEBSERVER_PORT="8500"

# Show help
function show_help {
    echo "Usage: $(basename "$0") [options]"
    echo ""
    echo "Options:"
    echo "  --username USERNAME           HPC username (default: $USERNAME)"
    echo "  --cluster CLUSTER             Cluster name (e.g. capella, alpha) (default: $CLUSTER)"
    echo "  --hpc-url URL                 Full HPC login URL (overrides --cluster)"
    echo "  --local-port PORT             Local port for forwarding (default: $LOCAL_PORT)"
    echo "  --webserver-port PORT         Default port for webserver"
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
        --webserver-port)
            WEBSERVER_PORT="$2"
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

# Derive URL if not provided
if [[ -z "$HPC_URL" ]]; then
    HPC_URL="login1.${CLUSTER}.hpc.tu-dresden.de"
fi

# Setup virtual environment if needed
if [[ ! -d $VENV_DIR ]] || [[ ! -e $VENV_ACTIVATE ]]; then
    nice_echo "Virtual Environment '$VENV_DIR' not found. Trying to create it..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_ACTIVATE"
    nice_echo "Installing modules"
    pip install streamlit
    nice_echo "Done installing modules"
fi

source "$VENV_ACTIVATE"

nice_echo "Starting smartproxy as a background job"
python3 smartproxy.py &

nice_echo "Starting backend server as a background job"
bash "$SCRIPT_DIR/start_backend_from_enterprise_cloud.sh" \
    --username "$USERNAME" \
    --cluster "$CLUSTER" \
    --hpc-url "$HPC_URL" \
    --local-port "$LOCAL_PORT" &

nice_echo "Waiting for localhost:$LOCAL_PORT..."
until nc -z localhost "$LOCAL_PORT"; do
    sleep 1
done
nice_echo "localhost:$LOCAL_PORT is reachable!"

nice_echo "Starting streamlit script $SCRIPT_DIR/app.py"

streamlit run "$SCRIPT_DIR/app.py" --server.port "$WEBSERVER_PORT" --server.address 0.0.0.0

wait
