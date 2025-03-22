#!/bin/bash

# EXAMPLE USAGE:
# See stat-214-gsi/computing/psc-instructions.md for guidance on how to do this on PSC.
# These are settings for a hypothetical cluster and probably won't work on PSC
# sbatch job.sh configs/default.yaml

#SBATCH --account=mth240012p
#SBATCH --job-name=lab2-autoencoder-embedding
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --time=00:30:00
#SBATCH -o test-embedding.out
#SBATCH -e test-embedding.out

python get_embedding.py configs/default.yaml checkpoints/cnn-default-epoch=009-final.ckpt
