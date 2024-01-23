#!/bin/bash

function run_script {
  prefix_layer_min=$1
  prefix_layer_max=$2
  task_name=$3

  export TASK_NAME=$task_name
  export DATASET_NAME="bbbp"
  export MODEL_TYPE="prefix"
  export DATASET_TYPE="binary classification"
  cuda=0
  seed=43
  metric="auroc"
  prefix=1
  n_tasks=1
  bs=64
  weight_decay=0.0
  lr=4e-3
  dropout=0.1
  psl=40
  epochs=100
  cv=1
  lightweight=0
  output_file="./${DATASET_NAME}-${TASK_NAME}-ablations-desc-${prefix_layer_min}-${prefix_layer_max}.txt"

  CUDA_VISIBLE_DEVICES=$cuda python3 ../../run.py \
    --model_name_or_path "clefourrier/pcqm4mv2_graphormer_base" \
    --task_name "$TASK_NAME" \
    --data_path "/path/to/datasets/" \
    --split scaffold-0 \
    --dataset_name $DATASET_NAME \
    --dataset_type "$DATASET_TYPE" \
    --n_tasks $n_tasks \
    --metric $metric \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size $bs \
    --learning_rate $lr \
    --num_train_epochs $epochs \
    --pre_seq_len $psl \
    --output_dir "/path/to/output/${DATASET_NAME}-${TASK_NAME}-${MODEL_TYPE}-${prefix_layer_min}-${prefix_layer_max}/" \
    --logging_dir "/path/to/logs/${DATASET_NAME}-${TASK_NAME}-${MODEL_TYPE}-${prefix_layer_min}-${prefix_layer_max}/" \
    --report_to "tensorboard" \
    --overwrite_output_dir 1 \
    --hidden_dropout_prob $dropout \
    --seed $seed \
    --save_strategy no \
    --evaluation_strategy epoch \
    --prefix $prefix \
    --weight_decay $weight_decay \
    --logging_strategy epoch \
    --lightweight $lightweight \
    --cv $cv \
    --prefix_layer_min $prefix_layer_min \
    --prefix_layer_max $prefix_layer_max \
    &> "$output_file"
}

# Define the range of prefix_layer_min and prefix_layer_max values
prefix_layer_min_values=(2 4 6 8 3 6)
prefix_layer_max_values=(12 12 12 12 12 12)
task_name="graph"

# Execute the script with different values of prefix_layer_min and prefix_layer_max
for i in "${!prefix_layer_min_values[@]}"; do
  prefix_layer_min=${prefix_layer_min_values[i]}
  prefix_layer_max=${prefix_layer_max_values[i]}
  run_script $prefix_layer_min $prefix_layer_max $task_name
done
