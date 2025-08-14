set -e

TASK_NAME=_webshop
FILTER_RATIO=0.5
MODEL=Qwen/Qwen3-4B-Instruct-2507

python -m recipe.webshop.main_webshop --config-name $TASK_NAME \
    model_path=$MODEL \
    system.CUDA_VISIBLE_DEVICES=\"0,1,2,3,4,5,6,7\" \
    actor_rollout_ref.rollout.rollout_filter_ratio=$FILTER_RATIO \
    trainer.experiment_name=$TASK_NAME-ppo-rolloutfilter$FILTER_RATIO-$MODEL \
    trainer.nnodes=1
    