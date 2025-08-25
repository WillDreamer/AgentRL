set -e

TASK_NAME=_webshop
REMARK=prompt_v5_T_5_Len_1k_VoidRe_1_ctx_nothink
FILTER_RATIO=0.25
MODEL=Qwen/Qwen3-4B

WANDB_API_KEY="ba70fcbc92808cc7a1750dd80ac3908295e6854f" # Modify your wandb key
# ============================ Preparation ============================
# Login to WandB (if API key is provided)
if [ "$WANDB_API_KEY" != "" ]; then
    wandb login --relogin $WANDB_API_KEY
    export WANDB_DIR=${SAVE_PATH}
fi

python -m recipe.webshop.main_webshop --config-name $TASK_NAME \
    model_path=$MODEL \
    system.CUDA_VISIBLE_DEVICES=\"0,1,2,3,4,5,6,7\" \
    actor_rollout_ref.rollout.rollout_filter_ratio=$FILTER_RATIO \
    trainer.experiment_name=$TASK_NAME-$REMARK-ppo-filter$FILTER_RATIO-$MODEL \
    trainer.nnodes=1 \
    trainer.rollout_data_dir=log_rollout \
    agent_proxy.max_turn=5 \
    custom_envs.WebShop.max_actions_per_traj=5 \
    actor_rollout_ref.rollout.response_length=1000 \
    actor_rollout_ref.rollout.max_model_len=20000 \
    actor_rollout_ref.rollout.max_num_batched_tokens=20000 \
    trainer.total_training_steps=400
    
