#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name "hessian"     # Name of the job 
#SBATCH --export=ALL             # Export all environment variables
#SBATCH --output "hessian.log"   # Log-file (important!)
#SBATCH --cpus-per-task=4        # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4000       # Memory to allocate in MB per allocated CPU core
#SBATCH --gres=gpu:1             # Number of GPU's
#SBATCH --time="1-00:00:00"      # Max execution time

cd ~/HESSIAN
mamba activate hessian

python3 models/train.py \
--model_size small \
--save_path /home/aeymael/HESSIAN/weights \
--save_freq 5 \
--data_path /scratch/users/aeymael/PlantDiseaseDataset \
--epochs 1000 \
--lr 3e-4 \
--train_prop 0.8 \
--device cuda \
--img_size 224 \
--batch_size 32 \
--max_samples 654084404 \
--wandb_mode online \
--optimizer AdamW \
--seed 42
