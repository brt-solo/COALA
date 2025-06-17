#!/bin/bash
#SBATCH --job-name=synthetic_cf_complex_large_weight
#SBATCH --mail-type=ALL
#SBATCH --mail-user=19bh19@queensu.ca
#SBATCH --qos=privileged

#SBATCH --partition=reserved
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=0-24:00:00
#SBATCH --output=synthetic_cf_complex_large_weight.out
#SBATCH --error=synthetic_cf_complex_large_weight.err

export PYTHONPATH=$PYTHONPATH:/global/home/hpc5434/multimodal/Spec/TabSS
export PYTHONPATH=$PYTHONPATH:/global/home/hpc5434/MAP-CF


cd /global/home/hpc5434/MAP-CF/synthetic
python -u synthetic_cf.py