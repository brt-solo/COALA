#!/bin/bash
#SBATCH --job-name=xgb_nhanes_diet_coala
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your@email.com
#SBATCH --qos=privileged
#SBATCH --partition=reserved
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=0-12:00:00
#SBATCH --output=xgb_nhanes_diet_coala.out
#SBATCH --error=xgb_nhanes_diet_coala.err

# Activate your virtualenv, e.g.: source ~/myenv/bin/activate
source ~/tflow/bin/activate

python -u run_coala_xgb_nhanes_diet.py
