#!/bin/bash
#SBATCH --job-name=run-clot-former
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=ampere
#SBATCH --exclude=gpu[017,018]
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@rutgers.edu
#SBATCH --output=slurm/slurm_%j.out
#SBATCH --error=slurm/slurm_%j.err

echo "Job started at: $(date)"

mkdir -p slurm

module purge
module use /projects/community/modulefiles
module load gcc/10.2.0/openmpi
module load cuda/12.1
module load git/2.35.1-ez82

# activate uv venv
source .venv/bin/activate
uv sync

# set SSL/TLS certificates
export SSL_CERT_FILE=$(uv run python -c "import certifi; print(certifi.where())")

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working Directory: $(pwd)"
echo "Python Path: $PYTHONPATH"


# get current git commit
echo "Current git commit: "
git rev-parse --short HEAD

# Check GPU availability
nvidia-smi

# Run training
echo "Starting SwinV2 finetuning..."
uv run python main.py

echo "Job completed at: $(date)"