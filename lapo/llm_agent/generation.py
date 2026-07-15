# Copyright 2026 LAPO Authors
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

import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.tracking import Tracking
import shutil
import requests
import math
import numpy as np
import torch.nn.functional as F
from verl.utils.reward_score import qa_em
from lapo.prompts import GOLD_ANSWER_PREFIX, GOLD_ANSWER_SUFFIX
from lapo.algorithm import replace_token_span

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3
    info_gain_type: str = "log_prob_diff"
    log_prob_micro_batch_size: int = 1
    log_prob_use_dynamic_bsz: bool = False
    log_prob_max_token_len_per_gpu: int = 8192
    temperature: float = 1.0
    use_counterfactual_ig: bool = False
    ig_eps: float = 1e-6
    disable_old_gt_ig: bool = True
    lapo_score_type: str = "logprob"
    lapo_score_direction: str = "backward"


GT_ANSWER_PREFIX = "<think>Now I can answer the question.</think>\n<answer>"
GT_ANSWER_SUFFIX = "</answer>"

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        responses_str = [resp.split('</search>')[0] + '</search>'
                 if '</search>' in resp 
                 else resp.split('</answer>')[0] + '</answer>'
                 if '</answer>' in resp 
                 else resp
                 for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""

        processed_obs_ids = []
        info_prefix = "\n\n<information>"
        info_suffix = "</information>\n\n"
        prefix_ids = self.tokenizer(info_prefix, add_special_tokens=False)['input_ids']
        suffix_ids = self.tokenizer(info_suffix, add_special_tokens=False)['input_ids']

        for obs in next_obs:
            obs_ids = self.tokenizer(obs, add_special_tokens=False)['input_ids']
            if len(obs_ids) <= self.config.max_obs_length:
                processed_obs_ids.append(obs_ids)
                continue

            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {len(obs_ids)} & {self.config.max_obs_length}")
            info_start = obs.find(info_prefix)
            info_end = obs.rfind(info_suffix)
            if info_start >= 0 and info_end >= info_start:
                content_start = info_start + len(info_prefix)
                content = obs[content_start:info_end]
                content_ids = self.tokenizer(content, add_special_tokens=False)['input_ids']
                content_budget = max(0, self.config.max_obs_length - len(prefix_ids) - len(suffix_ids))
                obs_ids = prefix_ids + content_ids[:content_budget] + suffix_ids
            else:
                obs_ids = obs_ids[:self.config.max_obs_length]
            processed_obs_ids.append(obs_ids)

        return self._pad_token_lists(processed_obs_ids, pad_to_left=False)

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _valid_token_ids(self, ids: torch.Tensor) -> List[int]:
        if ids.numel() == 0:
            return []
        return ids[ids != self.tokenizer.pad_token_id].detach().cpu().tolist()

    def _append_response_parts(self, response_parts: List[List[Dict[str, Any]]],
                               responses_ids: torch.Tensor,
                               responses_str: List[str],
                               next_obs_ids: torch.Tensor = None,
                               next_obs: List[str] = None,
                               is_search: List[int] = None) -> None:
        batch_size = len(response_parts)
        for i in range(batch_size):
            response_ids = self._valid_token_ids(responses_ids[i])
            if not response_ids:
                continue
            obs_ids = self._valid_token_ids(next_obs_ids[i]) if next_obs_ids is not None else []
            response_parts[i].append({
                'response_ids': response_ids,
                'response_text': responses_str[i],
                'obs_ids': obs_ids,
                'obs_text': '' if next_obs is None else next_obs[i],
                'is_search': bool(is_search[i]) if is_search is not None else None,
            })

    def _clip_span(self, span: Tuple[int, int], max_len: int) -> Tuple[int, int]:
        start, end = span
        start = max(0, min(int(start), max_len))
        end = max(start, min(int(end), max_len))
        return start, end

    def _build_right_side_from_parts(self, response_parts: List[List[Dict[str, Any]]]) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        batch_size = len(response_parts)
        per_sample_ids, per_sample_info_masked_ids = [], []
        metadata = {
            'turn_spans': [],
            'turn_texts': [],
            'turn_is_search': [],
            'information_spans': [],
            'final_turn_idx': [],
            'pred_answer': [],
            'response_text': [],
            'answer_token_start': [],
            'answer_token_end': [],
        }

        for parts in response_parts:
            ids, masked_ids = [], []
            turn_spans, turn_texts, turn_is_search, information_spans = [], [], [], []
            for part in parts:
                turn_start = len(ids)
                resp_ids = part['response_ids']
                ids.extend(resp_ids)
                masked_ids.extend(resp_ids)
                obs_ids = part.get('obs_ids', [])
                if obs_ids:
                    info_start = len(ids)
                    ids.extend(obs_ids)
                    masked_ids.extend([self.tokenizer.pad_token_id] * len(obs_ids))
                    information_spans.append((info_start, len(ids)))
                turn_spans.append((turn_start, len(ids)))
                turn_texts.append(part.get('response_text', ''))
                # Only executed searches with an environment observation are eligible
                # for leave-one-turn attribution. A search emitted during the forced
                # final generation has no retrieval observation and is excluded.
                turn_is_search.append(bool(part.get('is_search')) and bool(obs_ids))

            per_sample_ids.append(ids)
            per_sample_info_masked_ids.append(masked_ids)

            response_text = self.tokenizer.decode(ids, skip_special_tokens=True)
            pred_answer = qa_em.extract_last_answer(response_text)
            answer_start, answer_end = self._find_last_answer_content_token_span(response_text)
            final_turn_idx = None
            if answer_start is not None:
                for idx, (start, end) in enumerate(turn_spans):
                    if start <= answer_start < end:
                        final_turn_idx = idx
                        break

            metadata['turn_spans'].append(turn_spans)
            metadata['turn_texts'].append(turn_texts)
            metadata['turn_is_search'].append(turn_is_search)
            metadata['information_spans'].append(information_spans)
            metadata['final_turn_idx'].append(final_turn_idx)
            metadata['pred_answer'].append(pred_answer)
            metadata['response_text'].append(response_text)
            metadata['answer_token_start'].append(answer_start)
            metadata['answer_token_end'].append(answer_end)

        effective_len = max([len(ids) for ids in per_sample_ids] + [0])
        max_len = min(self.config.max_prompt_length, effective_len)
        if max_len == 0:
            empty = torch.empty((batch_size, 0), dtype=torch.long)
            return {'responses': empty, 'responses_with_info_mask': empty.clone()}, metadata

        pad_id = self.tokenizer.pad_token_id
        responses = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        responses_with_info_mask = torch.full((batch_size, max_len), pad_id, dtype=torch.long)

        for i, (ids, masked_ids) in enumerate(zip(per_sample_ids, per_sample_info_masked_ids)):
            clipped_ids = ids[:max_len]
            clipped_masked_ids = masked_ids[:max_len]
            if clipped_ids:
                responses[i, :len(clipped_ids)] = torch.tensor(clipped_ids, dtype=torch.long)
                responses_with_info_mask[i, :len(clipped_masked_ids)] = torch.tensor(clipped_masked_ids, dtype=torch.long)

            metadata['turn_spans'][i] = [
                self._clip_span(span, max_len) for span in metadata['turn_spans'][i]
                if self._clip_span(span, max_len)[0] < self._clip_span(span, max_len)[1]
            ]
            metadata['information_spans'][i] = [
                self._clip_span(span, max_len) for span in metadata['information_spans'][i]
                if self._clip_span(span, max_len)[0] < self._clip_span(span, max_len)[1]
            ]
            if (metadata['answer_token_start'][i] is not None and metadata['answer_token_start'][i] >= max_len) or \
               (metadata['answer_token_end'][i] is not None and metadata['answer_token_end'][i] > max_len):
                metadata['answer_token_start'][i] = None
                metadata['answer_token_end'][i] = None
                metadata['pred_answer'][i] = None
                metadata['final_turn_idx'][i] = None
            elif metadata['final_turn_idx'][i] is not None and metadata['final_turn_idx'][i] >= len(metadata['turn_spans'][i]):
                metadata['final_turn_idx'][i] = None

        return {'responses': responses, 'responses_with_info_mask': responses_with_info_mask}, metadata

    def _find_last_answer_content_token_span(self, response_text: str) -> Tuple[Any, Any]:
        matches = list(re.finditer(r'<answer>(.*?)</answer>', response_text, re.DOTALL))
        if not matches:
            return None, None
        match = matches[-1]
        char_start, char_end = match.start(1), match.end(1)
        encoding = self.tokenizer(response_text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = encoding['offset_mapping']
        token_start, token_end = None, None
        for idx, (start, end) in enumerate(offsets):
            if token_start is None and start >= char_start:
                token_start = idx
            if start < char_end and end > char_start:
                token_end = idx + 1
        if token_start is None:
            token_start = token_end
        return token_start, token_end

    def _pad_token_lists(self, token_lists: List[List[int]], pad_to_left: bool = False) -> torch.Tensor:
        max_len = max(len(x) for x in token_lists)
        pad_id = self.tokenizer.pad_token_id
        out = torch.full((len(token_lists), max_len), pad_id, dtype=torch.long)
        for i, ids in enumerate(token_lists):
            if not ids:
                continue
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            if pad_to_left:
                out[i, -len(ids):] = ids_tensor
            else:
                out[i, :len(ids)] = ids_tensor
        return out

    def _build_logprob_batch(self, contexts: List[List[int]], answer_ids: List[int]) -> DataProto:
        contexts = [ctx[-self.config.max_prompt_length:] if len(ctx) > self.config.max_prompt_length else ctx for ctx in contexts]
        prompts = self._pad_token_lists(contexts, pad_to_left=True)
        responses = self._pad_token_lists([answer_ids for _ in contexts], pad_to_left=False)
        input_ids = torch.cat([prompts, responses], dim=1)
        prompt_mask = self.tensor_fn.create_attention_mask(prompts)
        response_mask = self.tensor_fn.create_attention_mask(responses)
        attention_mask = torch.cat([prompt_mask, response_mask], dim=1)
        position_ids = self.tensor_fn.create_position_ids(attention_mask)
        output = DataProto.from_dict({
            'prompts': prompts,
            'responses': responses,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
        })
        output.meta_info.update({
            'micro_batch_size': self.config.log_prob_micro_batch_size,
            'max_token_len': self.config.log_prob_max_token_len_per_gpu,
            'temperature': self.config.temperature,
            'use_dynamic_bsz': self.config.log_prob_use_dynamic_bsz,
        })
        return output

    def _compute_log_prob_with_gpu_padding(self, logprob_batch: DataProto) -> DataProto:
        size_divisor = max(1, int(self.config.num_gpus))
        logprob_batch_padded, pad_size = pad_dataproto_to_divisor(logprob_batch, size_divisor)
        log_prob_output_padded = self.actor_rollout_wg.compute_log_prob(logprob_batch_padded)
        return unpad_dataproto(log_prob_output_padded, pad_size)

    def _counterfactual_gt_target(self, ground_truth: Any) -> Tuple[List[int], int, int]:
        gt_text = self._ground_truth_to_text(ground_truth).strip()
        if not gt_text:
            return [], 0, 0

        full_text = f"{GOLD_ANSWER_PREFIX}{gt_text}{GOLD_ANSWER_SUFFIX}"
        encoding = self.tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
        token_ids = encoding['input_ids']
        offsets = encoding['offset_mapping']
        gt_char_start = len(GOLD_ANSWER_PREFIX)
        gt_char_end = gt_char_start + len(gt_text)
        score_start, score_end = None, None
        for token_idx, (char_start, char_end) in enumerate(offsets):
            if score_start is None and char_end > gt_char_start:
                score_start = token_idx
            if char_start < gt_char_end and char_end > gt_char_start:
                score_end = token_idx + 1

        if score_start is None or score_end is None:
            return [], 0, 0
        return token_ids, score_start, score_end

    def _score_lapo_rows(self, log_prob_output: DataProto, row_score_spans: List[Tuple[int, int]],
                         score_type: str) -> Tuple[List[float], List[float]]:
        old_log_probs = log_prob_output.batch['old_log_probs'].detach().cpu()
        row_logprob_scores = []
        for row_idx, (score_start, score_end) in enumerate(row_score_spans):
            if score_start >= score_end:
                row_logprob_scores.append(float('nan'))
            else:
                row_logprob_scores.append(float(old_log_probs[row_idx, score_start:score_end].mean().item()))

        if score_type == "logprob":
            return row_logprob_scores, row_logprob_scores

        if score_type == "entropy" and 'entropy' in log_prob_output.batch:
            entropy = log_prob_output.batch['entropy'].detach().cpu().float()
            row_process_scores = []
            for row_idx, (score_start, score_end) in enumerate(row_score_spans):
                if score_start >= score_end:
                    row_process_scores.append(float('nan'))
                else:
                    span_entropy = entropy[row_idx, score_start:score_end]
                    row_process_scores.append(float(span_entropy.mean().item()) if span_entropy.numel() else 0.0)
            return row_logprob_scores, row_process_scores

        if 'logits' not in log_prob_output.batch:
            raise RuntimeError(f"LAPO score_type={score_type} requires logits from compute_log_prob.")

        logits = log_prob_output.batch['logits'].detach().cpu().float()
        row_process_scores = []
        for row_idx, (score_start, score_end) in enumerate(row_score_spans):
            if score_start >= score_end:
                row_process_scores.append(float('nan'))
                continue
            span_logits = logits[row_idx, score_start:score_end]
            if span_logits.numel() == 0:
                row_process_scores.append(0.0)
                continue
            if score_type == "entropy":
                logp = torch.log_softmax(span_logits, dim=-1)
                p = torch.softmax(span_logits, dim=-1)
                entropy = -torch.sum(p * logp, dim=-1)
                row_process_scores.append(float(entropy.mean().item()))
            else:
                row_process_scores.append(float('nan'))
        return row_logprob_scores, row_process_scores

    def _compute_lapo_raw_score(self, score_type: str, full_idx: int, minus_idx: int,
                                row_logprob_scores: List[float],
                                row_process_scores: List[float],
                                logits: torch.Tensor = None,
                                row_score_spans: List[Tuple[int, int]] = None) -> float:
        if score_type == "logprob":
            raw = row_logprob_scores[full_idx] - row_logprob_scores[minus_idx]
        elif score_type == "entropy":
            raw = row_process_scores[minus_idx] - row_process_scores[full_idx]
        elif score_type == "kl":
            if logits is None or row_score_spans is None:
                return 0.0
            score_start, score_end = row_score_spans[full_idx]
            minus_start, minus_end = row_score_spans[minus_idx]
            span_len = min(score_end - score_start, minus_end - minus_start)
            if span_len <= 0:
                return 0.0
            logits_full = logits[full_idx, score_start:score_start + span_len].float()
            logits_minus = logits[minus_idx, minus_start:minus_start + span_len].float()
            p_full = torch.softmax(logits_full, dim=-1)
            logp_full = torch.log_softmax(logits_full, dim=-1)
            logp_minus = torch.log_softmax(logits_minus, dim=-1)
            kl = torch.sum(p_full * (logp_full - logp_minus), dim=-1)
            raw = float(kl.mean().item()) if kl.numel() else 0.0
        else:
            raise ValueError(f"Unsupported LAPO score_type: {score_type}")
        return 0.0 if not np.isfinite(raw) else float(raw)

    def _compute_counterfactual_ig_rewards(self, final_output: DataProto, ground_truths: List[Any]) -> Dict[str, Any]:
        batch_size = final_output.batch['responses'].shape[0]
        turn_ig_rewards = [[] for _ in range(batch_size)]
        raw_ig_values = [[] for _ in range(batch_size)]
        raw_process_scores = [[] for _ in range(batch_size)]
        final_rewards = [0.0 for _ in range(batch_size)]
        ig_enabled = [0 for _ in range(batch_size)]
        ig_target_sources = ['none' for _ in range(batch_size)]
        full_target_log_probs = [None for _ in range(batch_size)]
        deleted_target_log_probs = [[] for _ in range(batch_size)]
        lapo_score_type = str(getattr(self.config, 'lapo_score_type', 'logprob')).lower()
        if lapo_score_type not in ('logprob', 'kl', 'entropy'):
            raise ValueError(f"Unsupported algorithm.lapo_score_type={lapo_score_type}. Expected logprob, kl, or entropy.")
        lapo_score_direction = str(getattr(self.config, 'lapo_score_direction', 'backward')).lower()
        if lapo_score_direction not in ('backward', 'forward'):
            raise ValueError(f"Unsupported algorithm.lapo_score_direction={lapo_score_direction}. Expected backward or forward.")

        if not self.config.use_counterfactual_ig or not ground_truths:
            return {
                'turn_ig_rewards': turn_ig_rewards,
                'raw_ig_values': raw_ig_values,
                'raw_process_scores': raw_process_scores,
                'lapo_score_type': [lapo_score_type for _ in range(batch_size)],
                'lapo_score_direction': [lapo_score_direction for _ in range(batch_size)],
                'final_rewards': final_rewards,
                'ig_enabled': ig_enabled,
                'ig_target_sources': ig_target_sources,
                'full_target_log_probs': full_target_log_probs,
                'deleted_target_log_probs': deleted_target_log_probs,
            }

        contexts, row_meta = [], []
        delete_ids = self.tokenizer("[DELETE]", add_special_tokens=False)['input_ids']
        prompt_len = final_output.batch['prompts'].shape[1]
        for i in range(batch_size):
            pred_answer = final_output.non_tensor_batch['pred_answer'][i]
            turn_spans = list(final_output.non_tensor_batch['turn_spans'][i])
            turn_texts = list(final_output.non_tensor_batch.get('turn_texts', [[] for _ in range(batch_size)])[i])
            turn_is_search = list(final_output.non_tensor_batch.get('turn_is_search', [[] for _ in range(batch_size)])[i])
            answer_start = final_output.non_tensor_batch['answer_token_start'][i]
            final_turn_idx = final_output.non_tensor_batch.get('final_turn_idx', np.array([None] * batch_size, dtype=object))[i]
            gt = ground_truths[i] if i < len(ground_truths) else None
            gt_target = gt.get('target', gt) if isinstance(gt, dict) else gt
            final_reward = qa_em.f1_check(pred_answer, gt_target)
            final_rewards[i] = final_reward
            turn_ig_rewards[i] = [0.0] * len(turn_spans)
            raw_ig_values[i] = [0.0] * len(turn_spans)
            raw_process_scores[i] = [0.0] * len(turn_spans)

            target_ids, score_start, score_end = self._counterfactual_gt_target(gt)
            ig_target_sources[i] = 'ground_truth'

            if not target_ids or score_start >= score_end or not turn_spans:
                continue

            valid_prompt_ids = self._valid_token_ids(final_output.batch['prompts'][i])
            valid_response_ids = self._valid_token_ids(final_output.batch['responses'][i])
            # The paper scores only the support from preceding search interactions and
            # excludes the complete policy-generated final-answer turn.
            target_insert_pos = len(valid_response_ids)
            if final_turn_idx is not None and 0 <= int(final_turn_idx) < len(turn_spans):
                target_insert_pos = int(turn_spans[int(final_turn_idx)][0])
            response_before_answer_ids = valid_response_ids[:target_insert_pos]
            context_full = valid_prompt_ids + response_before_answer_ids
            base_response_offset = len(valid_prompt_ids)
            answer_turn_idx = None
            if final_turn_idx is not None:
                answer_turn_idx = int(final_turn_idx)
            elif answer_start is not None:
                for idx, (start, end) in enumerate(turn_spans):
                    if int(start) <= int(answer_start) < int(end):
                        answer_turn_idx = idx
                        break

            sample_start = len(contexts)
            if lapo_score_direction == "backward":
                contexts.append(context_full)
            process_turn_indices = []
            for turn_idx, (start, end) in enumerate(turn_spans):
                if answer_turn_idx is not None and turn_idx == answer_turn_idx:
                    continue
                if turn_idx < len(turn_is_search) and turn_is_search[turn_idx] is not None:
                    is_search_turn = bool(turn_is_search[turn_idx])
                elif turn_idx < len(turn_texts):
                    action, _ = self.postprocess_predictions([str(turn_texts[turn_idx])])
                    is_search_turn = action[0] == 'search'
                else:
                    is_search_turn = True
                if not is_search_turn:
                    continue
                span_start = max(0, min(int(start), target_insert_pos))
                span_end = max(span_start, min(int(end), target_insert_pos))
                if lapo_score_direction == "backward":
                    if span_start >= span_end:
                        context_minus = list(context_full)
                    else:
                        ctx_start = base_response_offset + span_start
                        ctx_end = base_response_offset + span_end
                        context_minus = replace_token_span(
                            context_full, ctx_start, ctx_end, delete_ids
                        )
                    contexts.append(context_minus)
                else:
                    context_before = valid_prompt_ids + response_before_answer_ids[:span_start]
                    context_after = valid_prompt_ids + response_before_answer_ids[:span_end]
                    contexts.extend([context_before, context_after])
                process_turn_indices.append(turn_idx)
            if lapo_score_direction == "forward" and not process_turn_indices:
                continue
            row_meta.append((i, sample_start, len(turn_spans), process_turn_indices, target_ids, score_start, score_end))

        if not row_meta:
            return {
                'turn_ig_rewards': turn_ig_rewards,
                'raw_ig_values': raw_ig_values,
                'raw_process_scores': raw_process_scores,
                'lapo_score_type': [lapo_score_type for _ in range(batch_size)],
                'lapo_score_direction': [lapo_score_direction for _ in range(batch_size)],
                'final_rewards': final_rewards,
                'ig_enabled': ig_enabled,
                'ig_target_sources': ig_target_sources,
                'full_target_log_probs': full_target_log_probs,
                'deleted_target_log_probs': deleted_target_log_probs,
            }

        max_answer_len = max(len(meta[4]) for meta in row_meta)
        if max_answer_len == 0:
            return {
                'turn_ig_rewards': turn_ig_rewards,
                'raw_ig_values': raw_ig_values,
                'raw_process_scores': raw_process_scores,
                'lapo_score_type': [lapo_score_type for _ in range(batch_size)],
                'lapo_score_direction': [lapo_score_direction for _ in range(batch_size)],
                'final_rewards': final_rewards,
                'ig_enabled': ig_enabled,
                'ig_target_sources': ig_target_sources,
                'full_target_log_probs': full_target_log_probs,
                'deleted_target_log_probs': deleted_target_log_probs,
            }

        with torch.no_grad():
            # All rows in a single compute call need the same response tensor width. Only each row's
            # answer-content span contributes to the score; fixed pseudo-answer template tokens do not.
            row_answers, row_score_spans = [], []
            expanded_meta = []
            for sample_idx, start, num_turns, process_turn_indices, answer_ids, score_start, score_end in row_meta:
                num_variants = len(process_turn_indices) + 1 if lapo_score_direction == "backward" else len(process_turn_indices) * 2
                for _ in range(num_variants):
                    row_answers.append(answer_ids)
                    row_score_spans.append((score_start, score_end))
                expanded_meta.append((sample_idx, start, num_turns, process_turn_indices))
            prompts = self._pad_token_lists(
                [ctx[-self.config.max_prompt_length:] if len(ctx) > self.config.max_prompt_length else ctx for ctx in contexts],
                pad_to_left=True,
            )
            responses = self._pad_token_lists(row_answers, pad_to_left=False)
            input_ids = torch.cat([prompts, responses], dim=1)
            attention_mask = torch.cat([
                self.tensor_fn.create_attention_mask(prompts),
                self.tensor_fn.create_attention_mask(responses),
            ], dim=1)
            logprob_batch = DataProto.from_dict({
                'prompts': prompts,
                'responses': responses,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'position_ids': self.tensor_fn.create_position_ids(attention_mask),
            })
            logprob_batch.meta_info.update({
                'micro_batch_size': self.config.log_prob_micro_batch_size,
                'max_token_len': self.config.log_prob_max_token_len_per_gpu,
                'temperature': self.config.temperature,
                'use_dynamic_bsz': self.config.log_prob_use_dynamic_bsz,
            })
            if lapo_score_type == 'kl':
                logprob_batch.meta_info['return_logits_for_lapo'] = True
            elif lapo_score_type == 'entropy':
                logprob_batch.meta_info['return_entropy_for_lapo'] = True
            log_prob_output = self._compute_log_prob_with_gpu_padding(logprob_batch)

        row_logprob_scores, row_scores = self._score_lapo_rows(log_prob_output, row_score_spans, lapo_score_type)
        logits = log_prob_output.batch['logits'].detach().cpu().float() if lapo_score_type == "kl" else None

        for sample_idx, start, num_turns, process_turn_indices in expanded_meta:
            if lapo_score_direction == "backward":
                s_full = row_logprob_scores[start]
            elif process_turn_indices:
                s_full = row_logprob_scores[start + len(process_turn_indices) * 2 - 1]
            else:
                s_full = float('nan')
            full_target_log_probs[sample_idx] = s_full
            deleted_scores = [s_full] * num_turns
            raw_scores = [0.0] * num_turns
            for offset, turn_idx in enumerate(process_turn_indices):
                if lapo_score_direction == "backward":
                    full_idx = start
                    baseline_idx = start + 1 + offset
                else:
                    baseline_idx = start + 2 * offset
                    full_idx = baseline_idx + 1
                s_minus = row_logprob_scores[baseline_idx]
                raw = self._compute_lapo_raw_score(
                    lapo_score_type,
                    full_idx=full_idx,
                    minus_idx=baseline_idx,
                    row_logprob_scores=row_logprob_scores,
                    row_process_scores=row_scores,
                    logits=logits,
                    row_score_spans=row_score_spans,
                )
                deleted_scores[turn_idx] = s_minus
                raw_scores[turn_idx] = raw
            deleted_target_log_probs[sample_idx] = deleted_scores
            raw_array = np.array(raw_scores, dtype=np.float32)
            # Group-level scaling is performed once in RewardManager.  Keeping only raw
            # gains here avoids an accidental, non-paper per-trajectory normalization.
            turn_ig_rewards[sample_idx] = raw_array.tolist()
            raw_ig_values[sample_idx] = raw_array.tolist()
            raw_process_scores[sample_idx] = raw_array.tolist()
            ig_enabled[sample_idx] = int(np.any(np.abs(raw_array) > self.config.ig_eps))

        total_variants = len(row_score_spans)
        total_rewards = sum(len(rewards) for rewards in turn_ig_rewards)
        print(f"[Search-R1 Counterfactual IG] vectorized {lapo_score_direction}/{lapo_score_type}: {len(row_meta)} samples, {total_variants} variants, {total_rewards} turn rewards, 1 compute_log_prob call")
        return {
            'turn_ig_rewards': turn_ig_rewards,
            'raw_ig_values': raw_ig_values,
            'raw_process_scores': raw_process_scores,
            'lapo_score_type': [lapo_score_type for _ in range(batch_size)],
            'lapo_score_direction': [lapo_score_direction for _ in range(batch_size)],
            'final_rewards': final_rewards,
            'ig_enabled': ig_enabled,
            'ig_target_sources': ig_target_sources,
            'full_target_log_probs': full_target_log_probs,
            'deleted_target_log_probs': deleted_target_log_probs,
        }

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        # Agent rollouts recompute old_log_probs once on the final composed trajectory.
        # Per-turn generation log-probs are discarded, so skip that extra actor forward.
        active_batch.meta_info['recompute_log_prob'] = False
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        padded_active_batch.meta_info.update(active_batch.meta_info)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def _ground_truth_to_text(self, ground_truth: Any) -> str:
        if isinstance(ground_truth, dict):
            target = ground_truth.get('target', ground_truth.get('answer', ground_truth))
        else:
            target = ground_truth
        if isinstance(target, np.ndarray):
            target = target.tolist()
        if isinstance(target, (list, tuple)):
            return str(target[0]) if target else ''
        return str(target)

    def _build_pseudo_gt_responses(self, ground_truths: List[Any]) -> Tuple[List[List[int]], List[List[int]]]:
        pseudo_responses = []
        gt_idx = []

        for ground_truth in ground_truths:
            gt_text = self._ground_truth_to_text(ground_truth).strip()
            full_text = f"{GT_ANSWER_PREFIX}{gt_text}{GT_ANSWER_SUFFIX}"
            encoding = self.tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
            token_ids = encoding['input_ids'].tolist()[0]
            offset_mapping = encoding['offset_mapping'].tolist()[0]
            pseudo_responses.append(token_ids)

            gt_char_start = len(GT_ANSWER_PREFIX)
            gt_char_end = gt_char_start + len(gt_text)
            gt_token_start = None
            gt_token_end = None
            for token_idx, (char_start, char_end) in enumerate(offset_mapping):
                if gt_token_start is None and char_end > gt_char_start:
                    gt_token_start = token_idx
                if char_start < gt_char_end and char_end > 0:
                    gt_token_end = token_idx + 1

            gt_idx.append([
                len(token_ids) if gt_token_start is None else gt_token_start,
                len(token_ids) if gt_token_end is None else gt_token_end,
            ])

        return pseudo_responses, gt_idx

    def _pseudo_generate_sequences(self, prompts: DataProto, responses: List[List[int]]) -> DataProto:
        response_tensor = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(resp, dtype=torch.long) for resp in responses],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).to(prompts.batch['input_ids'].device)

        input_ids = torch.cat([prompts.batch['input_ids'], response_tensor], dim=-1)
        response_attention_mask = self.tensor_fn.create_attention_mask(response_tensor)
        attention_mask = torch.cat([prompts.batch['attention_mask'], response_attention_mask], dim=-1)

        response_length = response_tensor.size(1)
        last_valid_pos = (prompts.batch['attention_mask'].sum(dim=1, keepdim=True).long() - 1).clamp(min=0)
        delta_position_id = torch.arange(1, response_length + 1, device=prompts.batch['position_ids'].device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(response_tensor.size(0), -1)
        response_position_ids = last_valid_pos + delta_position_id
        position_ids = torch.cat([prompts.batch['position_ids'], response_position_ids], dim=-1)

        output = DataProto.from_dict({
            'prompts': prompts.batch['input_ids'],
            'responses': response_tensor,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
        })
        output.meta_info.update(prompts.meta_info)
        output.meta_info.update({
            'micro_batch_size': self.config.log_prob_micro_batch_size,
            'max_token_len': self.config.log_prob_max_token_len_per_gpu,
            'temperature': self.config.temperature,
            'use_dynamic_bsz': self.config.log_prob_use_dynamic_bsz,
        })
        return output

    def _clone_dataproto_batch(self, data: DataProto) -> DataProto:
        output = DataProto.from_dict({
            key: value.clone() for key, value in data.batch.items()
        })
        output.meta_info.update(data.meta_info)
        return output

    def _prealign_single_turn(self, pseudo_output: DataProto, target_prompt_len: int) -> DataProto:
        prompts = pseudo_output.batch['prompts']
        responses = pseudo_output.batch['responses']
        attention_mask = pseudo_output.batch['attention_mask']
        position_ids = pseudo_output.batch['position_ids']

        batch_size = prompts.shape[0]
        prompt_len = prompts.shape[1]
        response_len = responses.shape[1]
        pad_len = target_prompt_len - prompt_len

        if pad_len <= 0:
            aligned = self._clone_dataproto_batch(pseudo_output)
        else:
            pad_id = self.tokenizer.pad_token_id
            aligned_prompts = F.pad(prompts, (pad_len, 0), value=pad_id)
            aligned_input_ids = torch.cat([aligned_prompts, responses], dim=1)

            prompt_mask = attention_mask[:, :prompt_len]
            response_mask = attention_mask[:, prompt_len:prompt_len + response_len]
            pad_mask = torch.zeros(batch_size, pad_len, dtype=attention_mask.dtype, device=attention_mask.device)
            aligned_attention_mask = torch.cat([pad_mask, prompt_mask, response_mask], dim=1)

            prompt_pos = position_ids[:, :prompt_len]
            response_pos = position_ids[:, prompt_len:prompt_len + response_len]
            pad_pos = torch.zeros(batch_size, pad_len, dtype=position_ids.dtype, device=position_ids.device)
            aligned_position_ids = torch.cat([pad_pos, prompt_pos, response_pos], dim=1)

            aligned = DataProto.from_dict({
                'prompts': aligned_prompts,
                'responses': responses.clone(),
                'input_ids': aligned_input_ids,
                'attention_mask': aligned_attention_mask,
                'position_ids': aligned_position_ids,
            })

        aligned.meta_info.update(pseudo_output.meta_info)
        return aligned

    def _merge_prealigned_turns(self, aligned_outputs: List[DataProto]) -> DataProto:
        merged = DataProto.from_dict({
            'prompts': torch.cat([o.batch['prompts'] for o in aligned_outputs], dim=0),
            'responses': torch.cat([o.batch['responses'] for o in aligned_outputs], dim=0),
            'input_ids': torch.cat([o.batch['input_ids'] for o in aligned_outputs], dim=0),
            'attention_mask': torch.cat([o.batch['attention_mask'] for o in aligned_outputs], dim=0),
            'position_ids': torch.cat([o.batch['position_ids'] for o in aligned_outputs], dim=0),
        })
        merged.meta_info.update(aligned_outputs[0].meta_info)
        return merged

    def _compute_vectorized_info_gain_rewards(self, vectorized_data: Dict[str, Any]) -> List[List[float]]:
        pseudo_outputs = vectorized_data['pseudo_outputs_per_turn']
        append_masks = vectorized_data['append_masks_per_turn']
        gt_idx = vectorized_data['gt_idx']
        batch_size = vectorized_data['batch_size']

        num_turns = len(pseudo_outputs)
        if num_turns == 0 or gt_idx is None:
            return [[] for _ in range(batch_size)]
        num_samples = len(gt_idx)
        info_gain_rewards = [[] for _ in range(num_samples)]

        max_prompt_len = max(output.batch['prompts'].shape[1] for output in pseudo_outputs)
        aligned_outputs = [
            self._prealign_single_turn(output, target_prompt_len=max_prompt_len)
            for output in pseudo_outputs
        ]
        merged_batch = self._merge_prealigned_turns(aligned_outputs)
        log_prob_output = self._compute_log_prob_with_gpu_padding(merged_batch)
        merged_old_log_probs = log_prob_output.batch['old_log_probs']

        gt_values = {}
        for turn_idx in range(num_turns):
            start_idx = turn_idx * num_samples
            turn_old_log_probs = merged_old_log_probs[start_idx:start_idx + num_samples]
            append_mask = append_masks[turn_idx]

            for sample_idx, idx_pair in enumerate(gt_idx):
                start, end = idx_pair
                if start >= end:
                    continue

                log_probs = turn_old_log_probs[sample_idx, start:end]
                value = self._value_from_log_probs(log_probs)
                if math.isnan(value) or math.isinf(value):
                    if sample_idx in gt_values and append_mask[sample_idx]:
                        info_gain_rewards[sample_idx].append(0.0)
                    continue

                should_append = bool(append_mask[sample_idx])
                should_update_baseline = (turn_idx == 0) or should_append

                if sample_idx in gt_values and should_append:
                    gain = value - gt_values[sample_idx]
                    info_gain_rewards[sample_idx].append(0.0 if math.isnan(gain) or math.isinf(gain) else gain)

                if should_update_baseline:
                    gt_values[sample_idx] = value

        total_rewards = sum(len(rewards) for rewards in info_gain_rewards)
        print(f"[Search-R1 IG] vectorized GT logprob: {num_turns} turns, {total_rewards} info-gain rewards, 1 compute_log_prob call")
        return info_gain_rewards

    def _collect_vectorized_gt_state(self, rollings: DataProto, ground_truths: List[Any], append_mask: List[int],
                                     vectorized_data: Dict[str, Any]) -> None:
        if not ground_truths:
            return
        pseudo_responses, gt_idx = self._build_pseudo_gt_responses(ground_truths)
        if vectorized_data['gt_idx'] is None:
            vectorized_data['gt_idx'] = gt_idx
        pseudo_output = self._pseudo_generate_sequences(rollings, pseudo_responses)
        vectorized_data['pseudo_outputs_per_turn'].append(self._clone_dataproto_batch(pseudo_output))
        vectorized_data['append_masks_per_turn'].append(list(append_mask))

    def _value_from_log_probs(self, log_probs: torch.Tensor) -> float:
        mean_log_prob = log_probs.mean().item()
        if math.isnan(mean_log_prob) or math.isinf(mean_log_prob):
            return math.nan
        if self.config.info_gain_type == "prob_diff":
            return math.exp(mean_log_prob)
        return mean_log_prob

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor, ground_truths: List[Any] = None) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch
        if ground_truths is None:
            ground_truths = []
        batch_size = gen_batch.batch['input_ids'].shape[0]
        response_parts = [[] for _ in range(batch_size)]
        vectorized_data = {
            'pseudo_outputs_per_turn': [],
            'append_masks_per_turn': [],
            'gt_idx': None,
            'batch_size': batch_size,
        }
        rollings.batch = self.tensor_fn.cut_to_effective_len(
            rollings.batch,
            keys=['input_ids', 'attention_mask', 'position_ids']
        )
        if not self.config.disable_old_gt_ig and not self.config.use_counterfactual_ig:
            self._collect_vectorized_gt_state(rollings, ground_truths, [0] * batch_size, vectorized_data)

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Execute in environment and process observations
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask
            )
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)
            self._append_response_parts(
                response_parts, responses_ids, responses_str, next_obs_ids, next_obs, is_search
            )
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
            if not self.config.disable_old_gt_ig and not self.config.use_counterfactual_ig:
                self._collect_vectorized_gt_state(rollings, ground_truths, is_search, vectorized_data)
            
        # final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # # Execute in environment and process observations
            _, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=False
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            
            self._append_response_parts(response_parts, responses_ids, responses_str, is_search=is_search)

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        original_right_side, turn_metadata = self._build_right_side_from_parts(response_parts)
        if self.config.use_counterfactual_ig:
            final_output = self._compose_final_output(original_left_side, original_right_side, meta_info, metadata=turn_metadata)
            cf_ig = self._compute_counterfactual_ig_rewards(final_output, ground_truths)
            final_output = self._attach_counterfactual_ig(final_output, cf_ig)
            return final_output

        info_gain_rewards = None
        if not self.config.disable_old_gt_ig:
            info_gain_rewards = self._compute_vectorized_info_gain_rewards(vectorized_data)
        return self._compose_final_output(original_left_side, original_right_side, meta_info, info_gain_rewards, metadata=turn_metadata)

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict,
                            info_gain_rewards: List[List[float]] = None,
                            metadata: Dict[str, Any] = None) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        non_tensors = {}
        if info_gain_rewards is not None:
            non_tensors['info_gain_rewards'] = np.array(info_gain_rewards, dtype=object)
        if metadata is not None:
            for key, value in metadata.items():
                non_tensors[key] = np.array(value, dtype=object)

        final_output = DataProto.from_dict(final_output, non_tensors=non_tensors)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def _attach_counterfactual_ig(self, final_output: DataProto, cf_ig: Dict[str, Any]) -> DataProto:
        for key, value in cf_ig.items():
            final_output.non_tensor_batch[key] = np.array(value, dtype=object)
        return final_output

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []
        
        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        if do_search:
            search_results = self.batch_search(search_queries)
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        else:
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                elif action == 'search':
                    next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                else:
                    next_obs.append(f'\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
            
        assert len(search_results) == 0
            
        return next_obs, dones, valid_action, is_search

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(search|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    def batch_search(self, queries: List[str] = None) -> str:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        results = self._batch_search(queries)['result']
        
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
