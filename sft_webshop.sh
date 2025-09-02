#!/bin/bash
set -x

DATA_HOME=verl

mkdir $DATA_HOME/data

# wget -O $DATA_HOME/data/sft_dataset.parquet https://huggingface.co/datasets/willamazon1/AgentRL-SFT/blob/main/train.parquet

TRAIN_DATA=$DATA_HOME/data/train_qwen3.parquet
EVAL_DATA=$DATA_HOME/data/train_qwen3.parquet
MODEL_PATH=Qwen/Qwen3-4B
SAVE_PATH=./

WANDB_API_KEY="ba70fcbc92808cc7a1750dd80ac3908295e6854f" # Modify your wandb key
# ============================ Preparation ============================
# Login to WandB (if API key is provided)
if [ "$WANDB_API_KEY" != "" ]; then
    wandb login --relogin $WANDB_API_KEY
    export WANDB_DIR=${SAVE_PATH}
fi

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_DATA \
    data.val_files=$EVAL_DATA \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=16384 \
    data.train_batch_size=16 \
    data.micro_batch_size_per_gpu=2 \
    model.partial_pretrain=$MODEL_PATH \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=multiturn-sft \
    trainer.experiment_name=sft-qwen-3-4b-instruct-sp2 \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=2 \
    trainer.default_hdfs_dir=null \
    ulysses_sequence_parallel_size=8 \
    use_remove_padding=true
