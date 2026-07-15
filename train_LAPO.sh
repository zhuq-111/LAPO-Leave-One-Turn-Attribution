#!/usr/bin/env bash
# Copyright 2026 LAPO Authors
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export DATA_DIR="${DATA_DIR:-data/data_4full}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WANDB_ENV_FILE="$SCRIPT_DIR/.wandb.env"
if [[ -f "$WANDB_ENV_FILE" ]]; then
    # Kept outside version control; this is inherited by Ray worker processes.
    source "$WANDB_ENV_FILE"
else
    echo "No $WANDB_ENV_FILE found; using console logging only."
fi

if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export WANDB_MODE=online
    TRAINER_LOGGER="['console','wandb']"
else
    export WANDB_MODE=disabled
    TRAINER_LOGGER="['console']"
fi

if [[ "${USE_WANDB_PROXY:-0}" != "1" ]]; then
    export HTTP_PROXY=
    export HTTPS_PROXY=
    export ALL_PROXY=
    export http_proxy=
    export https_proxy=
    export all_proxy=
    export NO_PROXY="${NO_PROXY:+$NO_PROXY,}api.wandb.ai,wandb.ai,*.wandb.ai"
    export no_proxy="${no_proxy:+$no_proxy,}api.wandb.ai,wandb.ai,*.wandb.ai"
fi

WAND_PROJECT="${WAND_PROJECT:-LAPO}"

export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-LAPO/backward-logprob-lapo-lambda0.5-turn3}"
# export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-3b-it-em
# set -x
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

# max_prompt_length = (config['training']['max_start_length'] + config['training']['max_response_length'] * (config['training']['max_turns'] - 1) + config['training']['max_obs_length'] * config['training']['max_turns'])
mkdir -p "$(dirname "$EXPERIMENT_NAME.log")"

# LAPO score examples:
#   algorithm.lapo_score_direction=backward algorithm.lapo_score_type=logprob
#   algorithm.lapo_score_direction=backward algorithm.lapo_score_type=kl
#   algorithm.lapo_score_direction=backward algorithm.lapo_score_type=entropy
#   algorithm.lapo_score_direction=forward algorithm.lapo_score_type=logprob
#   algorithm.lapo_score_direction=forward algorithm.lapo_score_type=kl
#   algorithm.lapo_score_direction=forward algorithm.lapo_score_type=entropy
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/valid.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    algorithm.use_counterfactual_ig=true \
    algorithm.lambda_ig=0.5 \
    algorithm.ig_eps=1e-6 \
    algorithm.use_lapo_turn_gate=true \
    algorithm.disable_old_gt_ig=true \
    algorithm.lapo_score_type=logprob \
    algorithm.lapo_score_direction=backward \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=$TRAINER_LOGGER \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=6 \
    trainer.total_training_steps=200 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=3 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee $EXPERIMENT_NAME.log
