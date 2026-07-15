#!/bin/bash
#SBATCH --job-name=xgb_fhs_coala_test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your@email.com
#SBATCH --qos=privileged
#SBATCH --partition=reserved
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=0-12:00:00
#SBATCH --output=xgb_fhs_coala_test.out
#SBATCH --error=xgb_fhs_coala_test.err

# Activate your virtualenv, e.g.: source ~/myenv/bin/activate
source ~/tflow/bin/activate

python -u run_coala_xgb_fhs.py
