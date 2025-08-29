set -e

TASK_NAME=_webshop
REMARK=prompt_v5_T_5_Len_500_VoidRe_1_ctx_notemplate_lr_5e-7_voidmask_budget200
FILTER_RATIO=0.5
MODEL=Qwen/Qwen3-0.6B
MODEL_SHORT="${MODEL##*/}"

ROLLOUT_MODE="sync"
if [ "$ROLLOUT_MODE" = "async" ]; then
    export VLLM_USE_V1=1
fi

WANDB_API_KEY="ba70fcbc92808cc7a1750dd80ac3908295e6854f" # Modify your wandb key
# ============================ Preparation ============================
# Login to WandB (if API key is provided)
if [ "$WANDB_API_KEY" != "" ]; then
    wandb login --relogin $WANDB_API_KEY
    export WANDB_DIR=${SAVE_PATH}
fi

python -m recipe.webshop.main_webshop --config-name $TASK_NAME \
    trainer.experiment_name="${TASK_NAME}-${REMARK}-ppo-filter${FILTER_RATIO}-${MODEL_SHORT}" \
    trainer.nnodes=1 \
    trainer.rollout_data_dir=log_rollout \
    system.CUDA_VISIBLE_DEVICES=\"0,1,2,3,4,5,6,7\" \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.rollout.rollout_filter_ratio=$FILTER_RATIO \
    agent_proxy.max_turn=9 \
    custom_envs.WebShop.max_actions_per_traj=9 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=$ROLLOUT_MODE \
    actor_rollout_ref.rollout.response_length=500 \
    actor_rollout_ref.rollout.max_model_len=15000 \
    actor_rollout_ref.rollout.max_num_batched_tokens=15000 \
    trainer.total_training_steps=400 \
    critic.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr=5e-7
    
