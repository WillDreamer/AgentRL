set -e

TASK_NAME=_webshop
FILTER_RATIO=0.5
REMARK=Pv5-Turn_9-Len_720-VoidRe_1-ctx_last_think-voidmask-budget_600-action_filter_${FILTER_RATIO}
MODEL=Qwen/Qwen3-4B
MODEL_SHORT="${MODEL##*/}"

ROLLOUT_MODE="sync"
if [ "$ROLLOUT_MODE" = "async" ]; then
    export VLLM_USE_V1=1
else
    export VLLM_USE_V1=0 
    ray status >/dev/null 2>&1 || ray start --head
fi

WANDB_API_KEY="ba70fcbc92808cc7a1750dd80ac3908295e6854f" # Modify your wandb key
# ============================ Preparation ============================
# Login to WandB (if API key is provided)
if [ "$WANDB_API_KEY" != "" ]; then
    wandb login --relogin $WANDB_API_KEY
    export WANDB_DIR=${SAVE_PATH}
fi

python -m recipe.webshop.main_webshop --config-name $TASK_NAME \
    trainer.experiment_name="${TASK_NAME}-${REMARK}-ppo-${MODEL_SHORT}" \
    trainer.nnodes=1 \
    trainer.rollout_data_dir=log_rollout \
    system.CUDA_VISIBLE_DEVICES=\"0,1,2,3,4,5,6,7\" \
    actor_rollout_ref.model.path=$MODEL \
    critic.model.path=$MODEL \
    model_path=$MODEL \
    actor_rollout_ref.rollout.rollout_filter_ratio=$FILTER_RATIO \
    agent_proxy.max_turn=9 \
    custom_envs.WebShop.max_actions_per_traj=9 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    critic.ppo_mini_batch_size=64 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=$ROLLOUT_MODE \
    actor_rollout_ref.rollout.response_length=720 \
    actor_rollout_ref.rollout.max_model_len=15000 \
    actor_rollout_ref.rollout.max_num_batched_tokens=15000 \
    trainer.total_training_steps=400 \
    
