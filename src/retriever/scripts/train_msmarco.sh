#!/bin/bash


model_to_train=jmvcoelho/pythia-160m-1024-marco-docs-bow-contrastive-pretrain
trained_model_save_path=./data/model
mkdir -p $trained_model_save_path

trained_model_name=pythia-160m-1024-marco-docs-bow-contrastive-pretrain-marco-passage-sft
training_data=Tevatron/msmarco-passage-aug
training_data_revision=8ec0b7a

output_dir=$trained_model_save_path/$trained_model_name

# Spot/preemptible instance: auto-resume if a checkpoint already exists.
# First run starts fresh; every restart after preemption continues from the latest checkpoint.
if ls -d "$output_dir"/checkpoint-* >/dev/null 2>&1; then
  resume_arg="--resume_from_checkpoint true"
  echo "Found existing checkpoint in $output_dir -> resuming training."
else
  resume_arg=""
  echo "No checkpoint found in $output_dir -> starting fresh."
fi

python -m driver.train \
  --output_dir $output_dir \
  --model_name_or_path $model_to_train \
  --dataset_name $training_data \
  --dataset_revision $training_data_revision \
  --save_strategy steps \
  --save_steps 0.2 \
  --save_total_limit 5 \
  $resume_arg \
  --bf16 \
  --gradient_checkpointing \
  --temperature 0.01 \
  --per_device_train_batch_size 128 \
  --train_group_size 10 \
  --learning_rate 1e-4 \
  --query_max_len 32 \
  --passage_max_len 128 \
  --num_train_epochs 1 \
  --logging_steps 1 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 2 \
  --report_to wandb \
  --run_name $trained_model_name