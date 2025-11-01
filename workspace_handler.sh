#!/bin/bash

# Farben
RED=$(tput setaf 1)
GREEN=$(tput setaf 2)
YELLOW=$(tput setaf 3)
CYAN=$(tput setaf 6)
BOLD=$(tput bold)
RESET=$(tput sgr0)

DEFAULT_USER=$(grep '^username' "$(cd "$(dirname "$0")" && pwd)/defaults.ini" | sed -e 's#.*=##')
DEFAULT_PARTITION=$(grep '^partition' "$(cd "$(dirname "$0")" && pwd)/defaults.ini" | sed -e 's#.*=##')

# Parameter
USERNAME=${1:-$DEFAULT_USER}
PARTITION=${2:-$DEFAULT_PARTITION}
NUM_DAYS=${3:-100}

echo "${CYAN}${BOLD}Using:${RESET} USER=$USERNAME PARTITION=$PARTITION"

# Remote ws_list abrufen
ws_list_output=$(ssh "$USERNAME@login1.$PARTITION.hpc.tu-dresden.de" "ws_list")
ws_list_exit_code=$?

if [[ $ws_list_exit_code -ne 0 ]]; then
    echo "${RED}${BOLD}ERROR:${RESET} ws_list failed with exit-code $ws_list_exit_code"
    exit 1
fi

if [[ -z "$ws_list_output" ]]; then
    echo "${RED}${BOLD}ERROR:${RESET} ws_list output was empty."
    exit 2
fi

# Alle faiss Workspaces holen
faiss_blocks=$(echo "$ws_list_output" | awk '/id: faiss/{flag=1;print;next}/id: faiss_/{flag=1;print;next}/^id:/{flag=0}flag')

if [[ -z "$faiss_blocks" ]]; then
    echo "${RED}${BOLD}ERROR:${RESET} No faiss workspaces found."
    exit 3
fi

# Neueste faiss-Variante bestimmen
latest_id=$(echo "$faiss_blocks" | grep "id:" | awk '{print $2}' | sed 's/faiss_//' | sort -n | tail -1)

if [[ "$latest_id" == "faiss" ]]; then
    current_id="faiss"
    current_index=0
else
    if [[ "$latest_id" =~ ^[0-9]+$ ]]; then
        current_id="faiss_$latest_id"
        current_index=$latest_id
    else
        current_id="faiss"
        current_index=0
    fi
fi

# Block vom neuesten Workspace extrahieren
current_block=$(echo "$faiss_blocks" | awk "/id: $current_id/{flag=1;print;next}/^id:/{flag=0}flag")

# Filesystem name
filesystem=$(echo "$current_block" | grep "filesystem name" | awk -F: '{print $2}' | xargs)
if [[ -z "$filesystem" ]]; then
    filesystem="horse"
fi

# Workspace directory
current_path=$(echo "$current_block" | grep "workspace directory" | awk -F: '{print $2}' | xargs)

# Available extensions
available_ext=$(echo "$current_block" | grep "available extensions" | awk -F: '{print $2}' | xargs)
if [[ -z "$available_ext" ]]; then
    available_ext=0
fi

# Remaining time
remaining_line=$(echo "$current_block" | grep "remaining time")
days=$(echo "$remaining_line" | grep -o '[0-9]\+ days' | awk '{print $1}')
hours=$(echo "$remaining_line" | grep -o '[0-9]\+ hours' | awk '{print $1}')

if [[ -z "$days" ]]; then days=0; fi
if [[ -z "$hours" ]]; then hours=0; fi
total_hours=$(( days * 24 + hours ))

echo "${CYAN}${BOLD}Current workspace:${RESET} $current_id"
echo "${CYAN}${BOLD}Filesystem:${RESET} $filesystem"
echo "${CYAN}${BOLD}Available extensions:${RESET} $available_ext"
echo "${CYAN}${BOLD}Remaining time:${RESET} $days days $hours hours"

# Fall 1: noch genug Zeit → nichts tun
if [[ $total_hours -ge 192 ]]; then
    echo "${GREEN}${BOLD}INFO:${RESET} Workspace lifetime is sufficient. No action required."
    exit 0
fi

# Fall 2: Weniger als 8 Tage → prüfen ob extend geht
if [[ $available_ext -gt 0 ]]; then
    echo "${YELLOW}${BOLD}Extending workspace:${RESET} $current_id"
    ssh "$USERNAME@login1.$PARTITION.hpc.tu-dresden.de" "ws_extend -F $filesystem $current_id $NUM_DAYS"
else
    new_index=$((current_index + 1))
    new_id="faiss_$new_index"

    echo "${YELLOW}${BOLD}No extensions left. Allocating new workspace:${RESET} $new_id"
    ssh "$USERNAME@login1.$PARTITION.hpc.tu-dresden.de" "ws_allocate -F $filesystem $new_id $NUM_DAYS"

    ws_list_after=$(ssh "$USERNAME@login1.$PARTITION.hpc.tu-dresden.de" "ws_list")
    new_path=$(echo "$ws_list_after" | awk "/id: $new_id/{flag=1;print;next}/^id:/{flag=0}flag" | grep "workspace directory" | awk -F: '{print $2}' | xargs)

    if [[ -z "$new_path" ]]; then
        echo "${RED}${BOLD}ERROR:${RESET} Could not determine path of new workspace."
        exit 4
    fi

    echo "${GREEN}${BOLD}Copying data from old to new workspace:${RESET}"
    ssh "$USERNAME@login1.$PARTITION.hpc.tu-dresden.de" "dtcp -r '$current_path/.' $new_path/"
fi
