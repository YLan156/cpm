#!/usr/bin/env bash
#SBATCH --job-name=emotion-goal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.log
# Add your actual --partition / --account via sbatch arguments.
# Cluster-specific GPU resource syntax or memory request may need adjustment.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the bundle root}"
source .venv/bin/activate
bash scripts/run_multi.sh
