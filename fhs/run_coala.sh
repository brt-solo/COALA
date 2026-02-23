#!/bin/bash -l
#SBATCH --job-name=fhs_tabpfn_multi_1000iter
#SBATCH --mail-type=ALL
#SBATCH --mail-user=19bh19@queensu.ca
#SBATCH --qos=privileged
#SBATCH --partition=gpu
#SBATCH --gres gpu:1
#SBATCH --time=0-36:00:00
#SBATCH --output=fhs_tabpfn_multi_1000iter.out
#SBATCH --error=fhs_tabpfn_multi_1000iter.err

module purge
module load python/3.11.5
source ~/tflow/bin/activate

export PYTHONPATH=$PYTHONPATH:/global/home/hpc5434/MAP-CF


cd /global/home/hpc5434/MAP-CF/fhs

FEATURE_CAT=~/MAP-CF/fhs/feature_categories_multi_fhs.json
METHODS=("uniform")
MUT_RATES=("None")
INIT_POPS=(1000)
ITERS=( 10000)

MODELS=("tabpfn_fhs")

for MODEL_NAME in "${MODELS[@]}"; do

  if [ "$MODEL_NAME" == "perfect_model" ]; then
    MODEL=~/MAP-CF/synthetic/perfect_model.pkl
    REFERENCE=~/MAP-CF/synthetic/synthetic_test.csv
    TRAIN=~/MAP-CF/synthetic/synthetic_train.csv
  else
    MODEL=/global/home/hpc5434/models/tabpfn_fhs.pkl
    REFERENCE=~/MAP-CF/fhs/X_test_tabpfn_fhs.csv
    TRAIN=~/MAP-CF/fhs/X_train_tabpfn_fhs.csv
  fi

  for METHOD in "${METHODS[@]}"; do
    for RATE in "${MUT_RATES[@]}"; do
      for INIT in "${INIT_POPS[@]}"; do
        for ITER in "${ITERS[@]}"; do

          OUTPUT="${MODEL_NAME}_${METHOD}_init${INIT}_iter${ITER}_multi_fhs"
          if [ "$RATE" != "None" ]; then
            OUTPUT="${OUTPUT}_mut${RATE}"
          fi

          CMD="python -u run_coala.py \
            --model $MODEL \
            --reference $REFERENCE \
            --train $TRAIN \
            --feature_cat \"$FEATURE_CAT\" \
            --model_name $MODEL_NAME \
            --method $METHOD \
            --init_pop $INIT \
            --iter $ITER \
            --output $OUTPUT"

          if [ "$RATE" != "None" ]; then
            CMD="$CMD --mutation_rate $RATE"
          fi

          echo "Running: $CMD"
          start_time=$(date +%s)

          eval $CMD

          end_time=$(date +%s)
          elapsed=$((end_time - start_time))
          echo "Completed $OUTPUT in $elapsed seconds"
          echo ""


        done
      done
    done
  done
done
