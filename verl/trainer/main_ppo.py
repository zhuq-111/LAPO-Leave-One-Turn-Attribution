# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Modifications Copyright 2026 LOTAPO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np
from lotapo.algorithm import (
    add_process_advantage,
    build_process_advantages,
    standardize_outcomes,
)


def _char_pos_to_token_idx(char_pos, offset_mapping):
    for idx, (start, end) in enumerate(offset_mapping):
        if start <= char_pos < end:
            return idx
        if char_pos < start:
            return max(0, idx - 1)
    return max(0, len(offset_mapping) - 1)

def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em.compute_score_em
    else:
        raise NotImplementedError


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, format_score=0., algorithm_config=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score
        self.algorithm_config = algorithm_config
        self.use_counterfactual_ig = bool(getattr(algorithm_config, 'use_counterfactual_ig', False)) if algorithm_config is not None else False
        self.ig_eps = float(getattr(algorithm_config, 'ig_eps', 1e-6)) if algorithm_config is not None else 1e-6
        self.use_lotapo_turn_gate = bool(getattr(algorithm_config, 'use_lotapo_turn_gate', True)) if algorithm_config is not None else True
        self.lotapo_score_type = str(getattr(algorithm_config, 'lotapo_score_type', 'logprob')) if algorithm_config is not None else 'logprob'
        self.lotapo_score_direction = str(getattr(algorithm_config, 'lotapo_score_direction', 'backward')) if algorithm_config is not None else 'backward'

    def __call__(self, data: DataProto, info_gain_rewards=None):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        if self.use_counterfactual_ig:
            return self._call_counterfactual_ig(data)

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # all_scores = []

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            if valid_response_length <= 0:
                continue

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=self.format_score)

            reward_tensor[i, valid_response_length - 1] = score
            item_info_gain_rewards = []
            if info_gain_rewards is not None:
                item_info_gain_rewards = info_gain_rewards[i]
            elif 'info_gain_rewards' in data.non_tensor_batch:
                item_info_gain_rewards = data_item.non_tensor_batch.get('info_gain_rewards', [])

            if item_info_gain_rewards is not None and len(item_info_gain_rewards) > 0:
                encoding = self.tokenizer(response_str, return_offsets_mapping=True, add_special_tokens=False)
                offset_mapping = encoding['offset_mapping']
                info_end_positions = [m.end() - 1 for m in re.finditer(r'</information>', response_str)]
                for gain_idx, gain in enumerate(item_info_gain_rewards):
                    if gain is None:
                        continue
                    try:
                        gain = float(gain)
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(gain):
                        gain = 0.0
                    if gain == 0.0:
                        gain = 1e-10
                    if len(offset_mapping) == 0:
                        continue
                    if gain_idx < len(info_end_positions):
                        token_idx = _char_pos_to_token_idx(info_end_positions[gain_idx], offset_mapping)
                    else:
                        token_idx = max(0, int(valid_response_length) - 1)
                    token_idx = min(token_idx, int(valid_response_length) - 1, reward_tensor.shape[1] - 1)
                    reward_tensor[i, token_idx] += gain
            # all_scores.append(score)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)
        
        # print(f"[DEBUG] all_scores: {all_scores}")
        # print(f"[DEBUG] all_scores shape: {np.array(all_scores).shape}")
        # print(f"[DEBUG] all_scores mean: {np.mean(all_scores)}")
        # print(f"[DEBUG] all_scores max: {np.max(all_scores)}")
        # print(f"[DEBUG] all_scores min: {np.min(all_scores)}")
        # print(f"[DEBUG] all_scores std: {np.std(all_scores)}")

        return reward_tensor

    def _span_mask(self, spans, seq_len, device):
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        for start, end in spans:
            start = max(0, min(int(start), seq_len))
            end = max(start, min(int(end), seq_len))
            if end > start:
                mask[start:end] = True
        return mask

    def _call_counterfactual_ig(self, data: DataProto):
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        response_len = data.batch['responses'].shape[-1]
        attention_mask = data.batch['attention_mask'][:, -response_len:].bool()
        info_mask = data.batch['info_mask'][:, -response_len:].bool() if 'info_mask' in data.batch else attention_mask
        device = reward_tensor.device
        batch_size = len(data)

        final_rewards = np.zeros(batch_size, dtype=np.float32)
        already_print_data_sources = {}
        for i in range(batch_size):
            data_item = data[i]
            pred_answer = data_item.non_tensor_batch.get('pred_answer', None)
            if pred_answer is None:
                response_ids = data_item.batch['responses']
                valid_response_length = int(data_item.batch['attention_mask'][-response_len:].sum().item())
                pred_answer = qa_em.extract_last_answer(self.tokenizer.decode(response_ids[:valid_response_length]))
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            gt_target = ground_truth.get('target', ground_truth) if isinstance(ground_truth, dict) else ground_truth
            final_rewards[i] = qa_em.f1_check(pred_answer, gt_target)

            data_source = data_item.non_tensor_batch['data_source']
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                prompt_len = data_item.batch['prompts'].shape[-1]
                valid_prompt_len = int(data_item.batch['attention_mask'][:prompt_len].sum().item())
                valid_response_len = int(data_item.batch['attention_mask'][prompt_len:].sum().item())
                text = self.tokenizer.decode(torch.cat([
                    data_item.batch['prompts'][-valid_prompt_len:],
                    data_item.batch['responses'][:valid_response_len],
                ]))
                print(text)

        raw_ig_rows = data.non_tensor_batch.get('raw_ig_values', np.array([[]] * batch_size, dtype=object))
        raw_ig_arrays = []
        eligible_turn_rows = []
        signed_ig_rows = [None for _ in range(batch_size)]
        normalized_ig_rows = [None for _ in range(batch_size)]
        ig_tau_values = np.zeros(batch_size, dtype=np.float32)
        total_rewards = final_rewards.astype(np.float32).copy()
        lambda_ig = float(getattr(self.algorithm_config, 'lambda_ig', 1.0)) if self.algorithm_config is not None else 1.0
        uid = data.non_tensor_batch.get('uid', np.arange(batch_size).astype(object))

        for i in range(batch_size):
            raw_ig_arrays.append(np.array([float(x) for x in raw_ig_rows[i]], dtype=np.float32))
            turn_flags = list(data.non_tensor_batch.get(
                'turn_is_search', np.array([[]] * batch_size, dtype=object)
            )[i])
            final_turn = data.non_tensor_batch.get(
                'final_turn_idx', np.array([None] * batch_size, dtype=object)
            )[i]
            eligible_turn_rows.append(np.array([
                bool(turn_flags[turn_idx])
                and (final_turn is None or turn_idx != int(final_turn))
                if turn_idx < len(turn_flags) else False
                for turn_idx in range(raw_ig_arrays[-1].size)
            ], dtype=bool))

        process = build_process_advantages(
            raw_ig_arrays, uid, self.ig_eps, eligible_rows=eligible_turn_rows
        )
        signed_ig_rows = process.bounded
        normalized_ig_rows = process.normalized
        ig_tau_values = process.scales
        sample_adv = standardize_outcomes(final_rewards, uid, self.ig_eps)

        ig_credit_rows = []
        ig_advantage_rows = []
        final_credit_weights = np.zeros(batch_size, dtype=np.float32)
        for i in range(batch_size):
            turn_spans = list(data.non_tensor_batch.get('turn_spans', np.array([[]] * batch_size, dtype=object))[i])
            valid_non_info = attention_mask[i] & info_mask[i]
            normalized_turn_rewards = np.array(normalized_ig_rows[i], dtype=np.float32)
            final_token_adv = float(sample_adv[i])
            if self.use_lotapo_turn_gate:
                process_turn_rewards = process.gated[i]
            else:
                process_turn_rewards = normalized_turn_rewards
            turn_weights = np.abs(process_turn_rewards)
            turn_advantages = add_process_advantage(final_token_adv, process_turn_rewards, lambda_ig)
            ig_credit_rows.append(turn_weights.astype(np.float32).tolist())
            ig_advantage_rows.append(turn_advantages.astype(np.float32).tolist())
            final_credit_weights[i] = final_token_adv

            # Every policy-generated token keeps the outcome advantage.  The final-answer
            # turn has zero process gain, so Eq. 14 naturally reduces to A_out there.
            for turn_idx, turn_advantage in enumerate(turn_advantages):
                if turn_idx >= len(turn_spans):
                    break
                try:
                    turn_token_adv = float(turn_advantage)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(turn_token_adv) or turn_token_adv == 0.0:
                    continue
                turn_mask = self._span_mask([turn_spans[turn_idx]], response_len, device) & valid_non_info
                if int(turn_mask.sum().item()) > 0:
                    reward_tensor[i, turn_mask] += turn_token_adv

        raw_score_rows = data.non_tensor_batch.get('raw_process_scores', data.non_tensor_batch.get('raw_ig_values', []))
        raw_scores = [float(x) for row in raw_score_rows for x in row]
        signed_igs = [float(x) for row in signed_ig_rows for x in row]
        normalized_igs = [float(x) for row in normalized_ig_rows for x in row]
        credit_weights = [float(x) for row in ig_credit_rows for x in row]
        ig_advantages = [float(x) for row in ig_advantage_rows for x in row]
        active_credit_weights = [x for x in credit_weights if x > self.ig_eps]
        final_adv_positive = sample_adv > self.ig_eps
        final_adv_negative = sample_adv < -self.ig_eps
        final_adv_neutral = ~(final_adv_positive | final_adv_negative)
        positive_ig_credit_flags = []
        negative_ig_credit_flags = []
        for row in ig_advantage_rows:
            for advantage in row:
                positive_ig_credit_flags.append(float(advantage) > self.ig_eps)
                negative_ig_credit_flags.append(float(advantage) < -self.ig_eps)
        target_sources = list(data.non_tensor_batch.get('ig_target_sources', []))
        valid_rewards = reward_tensor[attention_mask]
        data.meta_info['cf_ig_metrics'] = {
            'lotapo/raw_turn_score_mean': float(np.mean(raw_scores)) if raw_scores else 0.0,
            'lotapo/raw_turn_score_std': float(np.std(raw_scores)) if len(raw_scores) > 1 else 0.0,
            'lotapo/raw_turn_score_min': float(np.min(raw_scores)) if raw_scores else 0.0,
            'lotapo/raw_turn_score_max': float(np.max(raw_scores)) if raw_scores else 0.0,
            'lotapo/nonzero_raw_turn_score_ratio': float(np.mean([abs(x) > self.ig_eps for x in raw_scores])) if raw_scores else 0.0,
            'lotapo/signed_turn_ig_mean': float(np.mean(signed_igs)) if signed_igs else 0.0,
            'lotapo/signed_turn_ig_min': float(np.min(signed_igs)) if signed_igs else 0.0,
            'lotapo/signed_turn_ig_max': float(np.max(signed_igs)) if signed_igs else 0.0,
            'lotapo/signed_ig_positive_turn_ratio': float(np.mean([x > self.ig_eps for x in signed_igs])) if signed_igs else 0.0,
            'lotapo/signed_ig_negative_turn_ratio': float(np.mean([x < -self.ig_eps for x in signed_igs])) if signed_igs else 0.0,
            'lotapo/normalized_turn_ig_mean': float(np.mean(normalized_igs)) if normalized_igs else 0.0,
            'lotapo/normalized_turn_ig_min': float(np.min(normalized_igs)) if normalized_igs else 0.0,
            'lotapo/normalized_turn_ig_max': float(np.max(normalized_igs)) if normalized_igs else 0.0,
            'lotapo/credit_weight_mean': float(np.mean(credit_weights)) if credit_weights else 0.0,
            'lotapo/credit_active_ratio': float(np.mean([x > self.ig_eps for x in credit_weights])) if credit_weights else 0.0,
            'lotapo/active_credit_weight_mean': float(np.mean(active_credit_weights)) if active_credit_weights else 0.0,
            'lotapo/credit_advantage_mean': float(np.mean(ig_advantages)) if ig_advantages else 0.0,
            'lotapo/credit_advantage_min': float(np.min(ig_advantages)) if ig_advantages else 0.0,
            'lotapo/credit_advantage_max': float(np.max(ig_advantages)) if ig_advantages else 0.0,
            'lotapo/credit_advantage_positive_ratio': float(np.mean(positive_ig_credit_flags)) if positive_ig_credit_flags else 0.0,
            'lotapo/credit_advantage_negative_ratio': float(np.mean(negative_ig_credit_flags)) if negative_ig_credit_flags else 0.0,
            'lotapo/ig_enabled_ratio': float(np.array(data.non_tensor_batch.get('ig_enabled', np.zeros(batch_size))).astype(np.float32).mean()),
            'lotapo/use_turn_gate': float(self.use_lotapo_turn_gate),
            'lotapo/ground_truth_target_ratio': float(np.mean([x == 'ground_truth' for x in target_sources])) if target_sources else 0.0,
            'reward/outcome_f1_mean': float(final_rewards.mean()) if final_rewards.size else 0.0,
            'reward/outcome_f1_std': float(final_rewards.std()) if final_rewards.size > 1 else 0.0,
            'reward/outcome_f1_min': float(final_rewards.min()) if final_rewards.size else 0.0,
            'reward/outcome_f1_max': float(final_rewards.max()) if final_rewards.size else 0.0,
            'reward/sample_advantage_mean': float(sample_adv.mean()) if sample_adv.size else 0.0,
            'reward/sample_advantage_std': float(sample_adv.std()) if sample_adv.size > 1 else 0.0,
            'reward/sample_advantage_min': float(sample_adv.min()) if sample_adv.size else 0.0,
            'reward/sample_advantage_max': float(sample_adv.max()) if sample_adv.size else 0.0,
            'reward/sample_advantage_positive_ratio': float(final_adv_positive.mean()) if sample_adv.size else 0.0,
            'reward/sample_advantage_negative_ratio': float(final_adv_negative.mean()) if sample_adv.size else 0.0,
            'reward/sample_advantage_neutral_ratio': float(final_adv_neutral.mean()) if sample_adv.size else 0.0,
            'reward/token_reward_mean': float(valid_rewards.mean().item()) if valid_rewards.numel() else 0.0,
            'reward/token_reward_std': float(valid_rewards.std().item()) if valid_rewards.numel() > 1 else 0.0,
            'reward/token_reward_min': float(valid_rewards.min().item()) if valid_rewards.numel() else 0.0,
            'reward/token_reward_max': float(valid_rewards.max().item()) if valid_rewards.numel() else 0.0,
        }
        data.non_tensor_batch['final_rewards'] = np.array(final_rewards.tolist(), dtype=object)
        data.non_tensor_batch['total_rewards'] = np.array(total_rewards.tolist(), dtype=object)
        data.non_tensor_batch['sample_advantages'] = np.array(sample_adv.tolist(), dtype=object)
        data.non_tensor_batch['final_advantages'] = np.array(sample_adv.tolist(), dtype=object)
        data.non_tensor_batch['normalized_ig_rewards'] = np.array(normalized_ig_rows, dtype=object)
        data.non_tensor_batch['signed_ig_rewards'] = np.array([row.tolist() for row in signed_ig_rows], dtype=object)
        data.non_tensor_batch['ig_credit_weights'] = np.array(ig_credit_rows, dtype=object)
        data.non_tensor_batch['ig_advantages'] = np.array(ig_advantage_rows, dtype=object)
        data.non_tensor_batch['final_credit_weights'] = np.array(final_credit_weights.tolist(), dtype=object)
        data.non_tensor_batch['ig_tau_values'] = np.array(ig_tau_values.tolist(), dtype=object)
        return reward_tensor


import os

import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    ray_env_vars = {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}
    for env_key in (
        'WANDB_API_KEY',
        'WANDB_MODE',
        'WANDB_BASE_URL',
        'HTTP_PROXY',
        'HTTPS_PROXY',
        'ALL_PROXY',
        'http_proxy',
        'https_proxy',
        'all_proxy',
        'NO_PROXY',
        'no_proxy',
    ):
        if env_key in os.environ:
            ray_env_vars[env_key] = os.environ[env_key]

    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': ray_env_vars})

    # Apply the environment to this task too, including when Ray was already initialized.
    ray.get(main_task.options(runtime_env={'env_vars': ray_env_vars}).remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, algorithm_config=config.algorithm)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
