#!/bin/bash -l
#SBATCH --job-name=tabpfn_shap_unscaled_k3
#SBATCH --mail-type=ALL
#SBATCH --mail-user=19bh19@queensu.ca
#SBATCH --qos=privileged
#SBATCH --partition=gpu
#SBATCH --gres gpu:1
#SBATCH --time=0-36:00:00
#SBATCH --output=tabpfn_shap_unscaled_k3.out
#SBATCH --error=tabpfn_shap_unscaled_k3.err

module purge
module load python/3.11.5
source ~/tflow/bin/activate

python -u tabpfn_shap.py