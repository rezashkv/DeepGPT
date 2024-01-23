export TASK_NAME=graph
export DATASET_NAME="toxcast"
export MODEL_TYPE="lightweight"
export DATASET_TYPE="multitask"

cuda=0
seed=43
metric="auroc"
prefix=0
n_tasks=617
bs=32
weight_decay=0.0
lr=3e-4
dropout=0.1
psl=38
epochs=100
cv=1
lightweight=1

CUDA_VISIBLE_DEVICES=$cuda python3 ../run.py \
  --model_name_or_path "clefourrier/pcqm4mv2_graphormer_base" \
  --task_name $TASK_NAME \
  --data_path "/path/to/KPGT/datasets/" \
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
  --output_dir /path/to/checkpoints/${DATASET_NAME}-${TASK_NAME}-${MODEL_TYPE}/ \
  --logging_dir /path/to/logs/${DATASET_NAME}-${TASK_NAME}-${MODEL_TYPE}/ \
  --report_to "tensorboard" \
  --overwrite_output_dir 1 \
  --hidden_dropout_prob $dropout \
  --seed $seed \
  --save_strategy no \
  --evaluation_strategy epoch \
  --prefix $prefix \
  --weight_decay $weight_decay\
  --logging_strategy epoch \
  --lightweight $lightweight \
  --cv $cv

