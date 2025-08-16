# Default values
MAX_TURNS=5
TRAIN_BATCH_SIZE=4
VAL_SAMPLE_SIZE=4
N_VAL=4
VAL_TEMPERATURE=1.0
VAL_BEFORE_TRAIN=True
MAX_PROMPT_LENGTH=16000
MAX_RESPONSE_LENGTH=8000
MAX_OBS_LENGTH=256
PPO_MINI_BATCH_SIZE=128
PPO_MICRO_TOKEN=24000
TOTAL_EPOCHS=1
TRAIN_DATASET=("/workspace/AgentRL/datasets/simplelr_math_35/train")
VALID_DATASET=("/workspace/AgentRL/datasets/simplelr_math_35/test" "/workspace/AgentRL/datasets/deepscaler/aime" "/workspace/AgentRL/datasets/deepscaler/aime25")
ROLLOUT_GPU_MEMORY_UTIL=0.75
ACTOR_OPTIMIZER_OFFLOAD=False
ACTOR_PARAMETER_OFFLOAD=False
MODEL_NAME=Qwen/Qwen2.5-3B
SAVE_FREQ=20
TEST_FREQ=10
REMOVE_CLIP=True
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=2
REJECTION_SAMPLE=True
SP_SIZE=1
GRAD_CLIP=1.0
ACC_FILTER=0.0_1.0
START_CLIP_STEP=20
BALANCE_BATCH=True
TOOL_USE=True
BIASED_ADV=True
MASK_VOID_TURNS=True
OVERSAMPLE=3
VAL_ONLY=False
LOG_VAL_GENERATIONS=64
OUTPUT_ACC_TO_FILE=False
CONFIG_NAME=simpletir_trainer
NNODES=1
GPUS_PER_NODE=8
RESUME=False
PROJECT_NAME=simpletir_math

# if resume is True, then set resume_mode to auto
if [ "$RESUME" = "True" ]; then
  RESUME_MODE="auto"
  echo "Resume from latest trial"
else
  # if resume is False, then set resume_mode to disable
  if [ "$RESUME" = "False" ]; then
    RESUME_MODE="disable"
    echo "No Resume"
  else
    # resume is path
    RESUME_MODE="resume_path"
    RESUME_FROM_PATH="$RESUME"
    echo "Resume from $RESUME"
  fi
fi

generate_model_micro_token() {
  local model_name=$1
  
  if [ "$PPO_MICRO_TOKEN" = "null" ]; then
    # Extract the model size (e.g., 7B, 14B, 32B) using regex  
    if [[ $model_name =~ ([0-9]+)B ]]; then
      local model_size="${BASH_REMATCH[1]}"
      # echo "Detected model size: ${model_size}B"
      
      # Set the basic config based on model size
      local micro_token_config
      case $model_size in
        3)
          micro_token_config=16384
          ;;
        7)
          micro_token_config=8192
          ;;
        14)
          micro_token_config=4096
          ;;
        24)
          micro_token_config=3072
          ;;
        32)
          micro_token_config=2048
          ;;
        *)
          # Unknown model size: ${model_size}B. Using default config of 16384
          micro_token_config=16384
          ;;
      esac

      # if you use tensor parallel, you can increase the micro token number
      if [ "$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE" -gt 1 ]; then
          micro_token_config=$((micro_token_config * ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE))
      fi
      
      echo $micro_token_config
    else
      # must manually set the token number
      echo 16384
    fi
  else
    echo $PPO_MICRO_TOKEN
  fi
}

generate_suffix() {
  local suffix=""

  while [[ "$#" -gt 0 ]]; do
    case $1 in
      --train_batch_size) suffix+="_batch$2"; shift 2 ;;
      --max_prompt_length) suffix+="_maxpro$2"; shift 2 ;;
      --max_response_length) suffix+="_maxres$2"; shift 2 ;;
      --max_obs_length) suffix+="_maxres$2"; shift 2 ;;
      --ppo_mini_batch_size) suffix+="_ppomini$2"; shift 2 ;;
      --remove_clip) suffix+="_rmclip$2"; shift 2 ;;
      --max_turns) suffix+="_maxturn$2"; shift 2;;
      --stp_on_err) suffix+="_stperr$2"; shift 2 ;;
      --grad_clip) suffix+="_gradclip$2"; shift 2 ;;
      --acc_filter) suffix+="_accfilter$2"; shift 2 ;;
      --start_clip_step) suffix+="_startclip$2"; shift 2 ;;
      --balance_batch) suffix+="_balbatch$2"; shift 2 ;;
      --mask_void_turns) suffix+="_maskvoidturns$2"; shift 2 ;;
      --oversample) suffix+="_oversample$2"; shift 2 ;;
      *) shift ;;
    esac
  done

  echo "$suffix"
}

echo "Arguments received: $@"

SUFFIX=$(generate_suffix "$@")
RUN_NAME="$RUN_NAME$SUFFIX"
LOG_FILE_PATH=$LOG_PATH/$RUN_NAME.log

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
  echo "Processing: $1"
  case "$1" in
    --config_name) CONFIG_NAME="$2"; shift 2 ;;
    --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --val_sample_size) VAL_SAMPLE_SIZE="$2"; shift 2 ;;
    --max_prompt_length) MAX_PROMPT_LENGTH="$2"; shift 2 ;;
    --max_response_length) MAX_RESPONSE_LENGTH="$2"; shift 2 ;;
    --max_obs_length) MAX_OBS_LENGTH="$2"; shift 2 ;;
    --ppo_mini_batch_size) PPO_MINI_BATCH_SIZE="$2"; shift 2 ;;
    --ppo_micro_token) PPO_MICRO_TOKEN="$2"; shift 2 ;;
    --n_val) N_VAL="$2"; shift 2 ;;
    --val_temperature) VAL_TEMPERATURE="$2"; shift 2 ;;
    --rollout_gpu_memory_util) ROLLOUT_GPU_MEMORY_UTIL="$2"; shift 2 ;;
    --rollout_tp) ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE="$2"; shift 2 ;;
    --actor_optimizer_offload) ACTOR_OPTIMIZER_OFFLOAD="$2"; shift 2 ;;
    --actor_parameter_offload) ACTOR_PARAMETER_OFFLOAD="$2"; shift 2 ;;
    --total_epochs) TOTAL_EPOCHS="$2"; shift 2 ;;
    --save_freq) SAVE_FREQ="$2"; shift 2 ;;
    --test_freq) TEST_FREQ="$2"; shift 2 ;;
    --remove_clip) REMOVE_CLIP="$2"; shift 2 ;;
    --rejection_sample) REJECTION_SAMPLE="$2"; shift 2 ;;
    --sp_size) SP_SIZE="$2"; shift 2 ;;
    --train_dataset) TRAIN_DATASET=($2); shift 2 ;;
    --valid_dataset) VALID_DATASET=($2); shift 2 ;;
    --model_name) MODEL_NAME="$2"; shift 2 ;;
    --max_turns) MAX_TURNS="$2"; shift 2 ;;
    --grad_clip) GRAD_CLIP="$2"; shift 2 ;;
    --acc_filter) ACC_FILTER="$2"; shift 2 ;;
    --start_clip_step) START_CLIP_STEP="$2"; shift 2 ;;
    --balance_batch) BALANCE_BATCH="$2"; shift 2 ;;
    --tool_use) TOOL_USE="$2"; shift 2 ;;
    --mask_void_turns) MASK_VOID_TURNS="$2"; shift 2 ;;
    --oversample) OVERSAMPLE="$2"; shift 2 ;;
    --val_only) VAL_ONLY="$2"; shift 2 ;;
    --log_val_generations) LOG_VAL_GENERATIONS="$2"; shift 2 ;;
    --output_acc_to_file) OUTPUT_ACC_TO_FILE="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [ ${#TRAIN_DATASET[@]} -gt 0 ]; then
  for dataset in "${TRAIN_DATASET[@]}"; do
    train_dataset_str+="_$(echo $dataset | sed 's/\//_/g')"  # replace '/' to '_'
  done
fi

RUN_NAME+="_$MODEL_NAME"

echo "RUN_NAME: $RUN_NAME"
echo "LOG_FILE_PATH: $LOG_FILE_PATH"

echo "Training with the following parameters:"
echo "Train Batch Size: $TRAIN_BATCH_SIZE"
echo "Max Prompt Length: $MAX_PROMPT_LENGTH"
echo "Max Response Length: $MAX_RESPONSE_LENGTH"
echo "PPO Mini Batch Size: $PPO_MINI_BATCH_SIZE"
echo "Total Epochs: $TOTAL_EPOCHS"
echo "Model Name: $MODEL_NAME"
echo "Remove Clip: $REMOVE_CLIP"
echo "grad clip: $GRAD_CLIP"
echo "balance batch: $BALANCE_BATCH"
echo "Mask Void Turns: $MASK_VOID_TURNS"
echo "Oversample Multiplier: $OVERSAMPLE"

# set ppo micro token
PPO_MICRO_TOKEN=$(generate_model_micro_token "$MODEL_NAME")
echo "PPO_MICRO_TOKEN: $PPO_MICRO_TOKEN"
LOG_PROB_MICRO_TOKEN=$((PPO_MICRO_TOKEN * 2))
max_num_batched_tokens=$(expr $MAX_PROMPT_LENGTH + $MAX_RESPONSE_LENGTH + 1000)


parse_acc_filter() {
    local acc_filter="$1"
    ACC_FILTER_LOW=$(echo $acc_filter | cut -d'_' -f1)
    ACC_FILTER_HIGH=$(echo $acc_filter | cut -d'_' -f2)

    # if ACC_FILTER_LOW is 0.0, set ACC_FILTER to False
    if (( $(echo "$ACC_FILTER_LOW == 0.0" | bc -l) )); then
        ACC_FILTER=False
    else
        ACC_FILTER=True
    fi
    echo "ACC_FILTER: $ACC_FILTER"
    echo "ACC_FILTER_LOW: $ACC_FILTER_LOW"
    echo "ACC_FILTER_HIGH: $ACC_FILTER_HIGH"
}

parse_acc_filter "$ACC_FILTER"

format_dataset_paths() {
  local dataset=("$@")
  local formatted_paths=""

  for dataset_path in "${dataset[@]}"; do
    formatted_paths+='"'${DATA_PATH}'/'"$dataset_path"'.parquet",'
  done

  formatted_paths="${formatted_paths%,}"

  echo "[$formatted_paths]"
}

TRAIN_FILES=$(format_dataset_paths "${TRAIN_DATASET[@]}")
VALID_FILES=$(format_dataset_paths "${VALID_DATASET[@]}")
echo "TRAIN_FILES: $TRAIN_FILES"
echo "VALID_FILES: $VALID_FILES"

echo "CONFIG_NAME: $CONFIG_NAME"

# Example of using the variables
sleep 3

export SANDBOX_ENDPOINT=http://127.0.0.1:12345/faas/sandbox/
export WANDB_ENTITY="RL_Reasoning"
export WANDB_PROMPT_VERSION="simpletir"
export WANDB_PROJECT="${WANDB_PROMPT_VERSION}"

WANDB_API_KEY="ba70fcbc92808cc7a1750dd80ac3908295e6854f" # Modify your wandb key
# ============================ Preparation ============================
# Login to WandB (if API key is provided)
if [ "$WANDB_API_KEY" != "" ]; then
    wandb login --relogin $WANDB_API_KEY
    export WANDB_DIR=${SAVE_PATH}
fi

PYTHONUNBUFFERED=1 python -m recipe.simpletir.main_simpletir \
    --config-name $CONFIG_NAME \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VALID_FILES \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_sample_size=$VAL_SAMPLE_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.model.path=$MODEL_NAME \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MICRO_TOKEN \
    actor_rollout_ref.actor.grad_clip=$GRAD_CLIP \
    actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAMETER_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP_SIZE \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$LOG_PROB_MICRO_TOKEN \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTIL \
    actor_rollout_ref.rollout.val_kwargs.n=$N_VAL \
    actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMPERATURE \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$LOG_PROB_MICRO_TOKEN \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$SP_SIZE\
    trainer.rejection_sample=$REJECTION_SAMPLE \
    trainer.acc_filter=$ACC_FILTER \
    trainer.acc_filter_low=$ACC_FILTER_LOW \
    trainer.acc_filter_high=$ACC_FILTER_HIGH \
    trainer.start_clip_step=$START_CLIP_STEP \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.remove_clip=$REMOVE_CLIP \
    trainer.oversample_multiplier=$OVERSAMPLE \
    trainer.log_val_generations=$LOG_VAL_GENERATIONS \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    trainer.default_local_dir=$CHECKPOINT_PATH/$RUN_NAME \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.resume_mode=$RESUME_MODE \
    trainer.resume_from_path=$RESUME_FROM_PATH \
    trainer.balance_batch=$BALANCE_BATCH \
    agent.tool_use=$TOOL_USE \
    agent.max_turns=$MAX_TURNS \
    data.max_start_length=4096 \
    data.max_obs_length=$MAX_OBS_LENGTH \
    actor_rollout_ref.actor.mask_tool_output=$TOOL_USE \
    actor_rollout_ref.actor.mask_void_turns=$MASK_VOID_TURNS \
    +trainer.val_only=$VAL_ONLY \
    +trainer.output_acc_to_file=$OUTPUT_ACC_TO_FILE \
    | tee -a $LOG_FILE_PATH
