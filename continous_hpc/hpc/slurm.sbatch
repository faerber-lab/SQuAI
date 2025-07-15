#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --mem=1GB
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=slurm_runner

get_slurm_script () {
	if command -v scontrol >/dev/null 2>&1; then
		local logfile
		scontrol show job "$1" | awk -F'Command=' '/Command=/{print $2}'
	else
		echo "scontrol nicht verfügbar." >&2
		return 1
	fi
}

if [ -n "$SLURM_JOB_ID" ]; then
	SLURM_SCRIPT="$(get_slurm_script $SLURM_JOB_ID)"
	sbatch --dependency=afterany:$SLURM_JOB_ID "$SLURM_SCRIPT" "$@"
	bash $(dirname $SLURM_SCRIPT)/run $*
else
	bash run $*
fi
