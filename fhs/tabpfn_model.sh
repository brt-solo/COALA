#!/bin/bash
#SBATCH --job-name=tabpfn_nhane
#SBATCH --mail-type=ALL
#SBATCH --mail-user=19bh19@queensu.ca
#SBATCH --qos=privileged
#SBATCH --partition=gpu
#SBATCH --gres gpu:1
#SBATCH --time=0-36:00:00
#SBATCH --output=tabpfn_nhane.out
#SBATCH --error=tabpfn_nhane.err

source ~/tflow/bin/activate
module load cuda


python -u tabpfn_model.py