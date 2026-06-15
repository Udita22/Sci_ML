#!/bin/bash

#SBATCH --job-name=vit_cifar
#SBATCH --partition=low_2h_1gpu

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

#SBATCH --time=02:00:00

#SBATCH --output=vit_%j.out
#SBATCH --error=vit_%j.err

python3 train_transformer.py