#!/bin/bash
#SBATCH --job-name=synthetic_cf_paper
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your@email.com
#SBATCH --qos=privileged

#SBATCH --partition=reserved
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=0-24:00:00
#SBATCH --output=synthetic_cf_paper.out
#SBATCH --error=synthetic_cf_paper.err

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH=$PYTHONPATH:"$REPO_ROOT"

cd "$SCRIPT_DIR"

# Activate your virtualenv, e.g.: source ~/myenv/bin/activate

FEATURE_CAT="$SCRIPT_DIR/synthetic_feature_categories_action.json"
METHODS=("uniform")
MUT_RATES=("None")
INIT_POPS=(1000 5000 10000)
ITERS=(5000 25000 50000)

MODELS=("mlp_model")

for MODEL_NAME in "${MODELS[@]}"; do

  if [ "$MODEL_NAME" == "perfect_model" ]; then
    MODEL="$SCRIPT_DIR/perfect_model.pkl"
    REFERENCE="$SCRIPT_DIR/synthetic_test.csv"
    TRAIN="$SCRIPT_DIR/synthetic_train.csv"
  else
    MODEL="$SCRIPT_DIR/mlp_model.pth"
    REFERENCE="$SCRIPT_DIR/synthetic_test_scaled.csv"
    TRAIN="$SCRIPT_DIR/synthetic_train_scaled.csv"
  fi

  for METHOD in "${METHODS[@]}"; do
    for RATE in "${MUT_RATES[@]}"; do
      for INIT in "${INIT_POPS[@]}"; do
        for ITER in "${ITERS[@]}"; do

          OUTPUT="${MODEL_NAME}_${METHOD}_init${INIT}_iter${ITER}_paper"
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
          echo "Completed $OUTPUT in $elapsed seconds"
          echo ""


        done
      done
    done
  done
done
