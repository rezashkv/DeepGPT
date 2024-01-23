export TASK_NAME=light
export DATASET_NAME="bace"
export MODEL_TYPE="lightweight"
export DATASET_TYPE="binary classification"

cuda=0
seed=43
metric="auroc"
prefix=0
n_tasks=1
bs=128
weight_decay=0.0
lr=3e-4c
dropout=0.1
psl=20
epochs=20
cv=1
lightweight=1

CUDA_VISIBLE_DEVICES=$cuda python3 ../run.py \
  --model_name_or_path "path/to/KPGT/pretrained/base/base.pth" \
  --task_name $TASK_NAME \
  --data_path "path/to/datasets/" \
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
  --output_dir /path/to/results/pt2/checkpoints/${DATASET_NAME}-${TASK_NAME}-${MODEL_TYPE}/ \
  --logging_dir /path/to/logs/pt2/${DATASET_NAME}-${TASK_NAME}-${MODEL_TYPE}/ \
  --report_to "tensorboard" \
  --overwrite_output_dir 1 \
  --hidden_dropout_prob $dropout \
  --seed $seed \
  --save_strategy no \
  --evaluation_strategy epoch \
  --prefix $prefix \
  --logging_strategy epoch \
  --weight_decay $weight_decay \
  --lightweight $lightweight \
  --cv $cv

