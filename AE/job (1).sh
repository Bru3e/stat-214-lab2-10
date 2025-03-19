#!/bin/bash

# EXAMPLE USAGE:
# See stat-214-gsi/computing/psc-instructions.md for guidance on how to do this on PSC.
# These are settings for a hypothetical cluster and probably won't work on PSC
# sbatch job.sh configs/default.yaml

#SBATCH --account=mth240012p
#SBATCH --job-name=lab2-autoencoder
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1 
#SBATCH --cpus-per-task=5
#SBATCH -o test-3.out
#SBATCH -e test-3.err
#SBATCH --time=02:00:00

python run_autoencoder.py $1
