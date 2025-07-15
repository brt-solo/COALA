#!/bin/bash
#SBATCH --job-name=synthetic_cf_multicat
#SBATCH --mail-type=ALL
#SBATCH --mail-user=19bh19@queensu.ca
#SBATCH --qos=privileged

#SBATCH --partition=reserved
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=0-24:00:00
#SBATCH --output=synthetic_cf_multicat.out
#SBATCH --error=synthetic_cf_multicat.err

export PYTHONPATH=$PYTHONPATH:/global/home/hpc5434/MAP-CF


cd /global/home/hpc5434/MAP-CF/synthetic
#!/bin/bash

FEATURE_CAT=~/MAP-CF/synthetic/synthetic_feature_categories.json
METHODS=("uniform")
MUT_RATES=("None")
INIT_POPS=(1000 5000 10000)
ITERS=(10000 25000 50000)

MODELS=("perfect_model" "mlp_model")

for MODEL_NAME in "${MODELS[@]}"; do

  if [ "$MODEL_NAME" == "perfect_model" ]; then
    MODEL=~/MAP-CF/synthetic/perfect_model.pkl
    REFERENCE=~/MAP-CF/synthetic/synthetic_test.csv
    TRAIN=~/MAP-CF/synthetic/synthetic_train.csv
  else
    MODEL=~/MAP-CF/synthetic/mlp_model.pth
    REFERENCE=~/MAP-CF/synthetic/synthetic_test_scaled.csv
    TRAIN=~/MAP-CF/synthetic/synthetic_train_scaled.csv
  fi

  for METHOD in "${METHODS[@]}"; do
    for RATE in "${MUT_RATES[@]}"; do
      for INIT in "${INIT_POPS[@]}"; do
        for ITER in "${ITERS[@]}"; do

          OUTPUT="${MODEL_NAME}_${METHOD}_init${INIT}_iter${ITER}_multicat_setmaxmin"
          if [ "$RATE" != "None" ]; then
            OUTPUT="${OUTPUT}_mut${RATE}"
          fi

          CMD="python -u synthetic_cf.py \
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
          echo "⏱️ Completed $OUTPUT in $elapsed seconds"
          echo ""


        done
      done
    done
  done
done
