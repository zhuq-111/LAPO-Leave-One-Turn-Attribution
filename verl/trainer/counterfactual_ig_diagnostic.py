# Copyright 2026 LOTAPO Authors
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

"""Run Search-R1 rollouts and inspect Counterfactual IG without training."""

import html
import json
import random
import re
import statistics
from pathlib import Path

import hydra
import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from search_r1.llm_agent.generation import GenerationConfig, LLMGenerationManager
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.main_ppo import RewardManager
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.fs import copy_local_path_from_hdfs


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return [_jsonable(x) for x in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(x) for x in value]
    return value


def _init_actor_rollout(config):
    if config.actor_rollout_ref.actor.strategy != "fsdp":
        raise NotImplementedError("The diagnostic currently supports the FSDP worker used by this project.")

    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    pool_id = "diagnostic_pool"
    manager = ResourcePoolManager(
        resource_pool_spec={pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes},
        mapping={Role.ActorRollout: pool_id},
    )
    manager.create_resource_pool()
    pool = manager.get_resource_pool(Role.ActorRollout)
    actor_cls = RayClassWithInitArgs(
        cls=ray.remote(ActorRolloutRefWorker),
        config=config.actor_rollout_ref,
        role="actor_rollout",
    )
    colocated_cls = create_colocated_worker_cls({"actor_rollout": actor_cls})
    container = RayWorkerGroup(resource_pool=pool, ray_cls_with_init=colocated_cls)
    actor_rollout = container.spawn(prefix_set={"actor_rollout"})["actor_rollout"]
    actor_rollout.init_model()
    return actor_rollout, container


def _select_indices(dataset_size, cfg):
    if cfg.indices is not None:
        return [int(x) for x in cfg.indices]
    count = min(int(cfg.num_questions), dataset_size)
    if cfg.split == "random":
        return random.Random(int(cfg.seed)).sample(range(dataset_size), count)
    start = int(cfg.start_index)
    return list(range(start, min(start + count, dataset_size)))


def _decode_span(tokenizer, response_ids, span):
    start, end = int(span[0]), int(span[1])
    return tokenizer.decode(response_ids[start:end], skip_special_tokens=True)


def _records_from_batch(tokenizer, batch, output, token_rewards, rollout_log_probs, question_indices, n_rollouts):
    records = []
    response_len = output.batch["responses"].shape[1]
    response_attention = output.batch["attention_mask"][:, -response_len:].bool()
    info_mask = output.batch["info_mask"][:, -response_len:].bool()

    for i in range(len(output)):
        valid_len = int(response_attention[i].sum().item())
        response_ids = output.batch["responses"][i, :valid_len]
        prompt_ids = output.batch["prompts"][i]
        prompt_ids = prompt_ids[prompt_ids != tokenizer.pad_token_id]
        answer_start = output.non_tensor_batch["answer_token_start"][i]
        target_pos = valid_len if answer_start is None else min(int(answer_start), valid_len)
        turns = []
        spans = list(output.non_tensor_batch["turn_spans"][i])
        turn_texts = list(output.non_tensor_batch["turn_texts"][i])
        turn_is_search = list(output.non_tensor_batch.get("turn_is_search", [[] for _ in range(len(output))])[i])
        final_turn_idx = output.non_tensor_batch.get("final_turn_idx", np.array([None] * len(output), dtype=object))[i]
        answer_turn_idx = None if answer_start is None or final_turn_idx is None else int(final_turn_idx)
        if answer_turn_idx is not None and 0 <= answer_turn_idx < len(spans):
            target_pos = min(int(spans[answer_turn_idx][0]), valid_len)
        full_context_ids = torch.cat([prompt_ids, response_ids[:target_pos]])
        deleted_scores = list(output.non_tensor_batch["deleted_target_log_probs"][i])
        final_f1 = float(batch.non_tensor_batch["final_rewards"][i])
        total_reward = float(batch.non_tensor_batch.get("total_rewards", np.zeros(len(batch), dtype=object))[i])
        sample_advantage = float(batch.non_tensor_batch.get("sample_advantages", np.zeros(len(batch), dtype=object))[i])
        normalized_ig_rewards = list(batch.non_tensor_batch.get(
            "normalized_ig_rewards",
            output.non_tensor_batch.get("turn_ig_rewards", np.array([[] for _ in range(len(output))], dtype=object)),
        )[i])
        signed_ig_rewards = list(batch.non_tensor_batch.get(
            "signed_ig_rewards",
            output.non_tensor_batch.get("turn_ig_rewards", np.array([[] for _ in range(len(output))], dtype=object)),
        )[i])
        ig_credit_weights = list(batch.non_tensor_batch.get(
            "ig_credit_weights",
            np.array([[] for _ in range(len(output))], dtype=object),
        )[i])
        ig_advantages = list(batch.non_tensor_batch.get(
            "ig_advantages",
            np.array([[] for _ in range(len(output))], dtype=object),
        )[i])
        final_credit_weight = float(batch.non_tensor_batch.get("final_credit_weights", np.zeros(len(batch), dtype=object))[i])
        for turn_idx, span in enumerate(spans):
            start, end = int(span[0]), min(int(span[1]), valid_len)
            trainable = response_attention[i, start:end] & info_mask[i, start:end]
            info_only = response_attention[i, start:end] & ~info_mask[i, start:end]
            turn_ids = output.batch["responses"][i, start:end]
            is_answer_turn = answer_turn_idx is not None and turn_idx == answer_turn_idx
            if is_answer_turn:
                is_search_turn = False
            elif turn_idx < len(turn_is_search) and turn_is_search[turn_idx] is not None:
                is_search_turn = bool(turn_is_search[turn_idx])
            elif turn_idx < len(turn_texts):
                match = re.search(r"<(search|answer)>(.*?)</\1>", str(turn_texts[turn_idx]), re.DOTALL)
                is_search_turn = match is not None and match.group(1) == "search"
            else:
                is_search_turn = False
            delete_start = min(start, target_pos) if is_search_turn else target_pos
            delete_end = min(end, target_pos) if is_search_turn else target_pos
            delete_ids = torch.tensor(
                tokenizer("[DELETE]", add_special_tokens=False)["input_ids"],
                dtype=response_ids.dtype,
                device=response_ids.device,
            ) if is_search_turn else response_ids[:0]
            deleted_response_ids = torch.cat([
                response_ids[:delete_start], delete_ids, response_ids[delete_end:target_pos]
            ])
            deleted_context_ids = torch.cat([prompt_ids, deleted_response_ids])
            ig_score = float(normalized_ig_rewards[turn_idx]) if turn_idx < len(normalized_ig_rewards) else 0.0
            signed_ig = float(signed_ig_rewards[turn_idx]) if turn_idx < len(signed_ig_rewards) else ig_score
            ig_credit_weight = float(ig_credit_weights[turn_idx]) if turn_idx < len(ig_credit_weights) else 0.0
            ig_advantage = float(ig_advantages[turn_idx]) if turn_idx < len(ig_advantages) else 0.0
            if is_answer_turn:
                allocation_component = final_credit_weight
                allocation_kind = "final"
            elif is_search_turn:
                allocation_component = ig_advantage
                allocation_kind = "ig"
            else:
                allocation_component = sample_advantage
                allocation_kind = "outcome"
            trainable_rewards = token_rewards[i, start:end][trainable]
            allocated_advantage = float(trainable_rewards.mean().item()) if trainable_rewards.numel() else 0.0
            turns.append({
                "turn": turn_idx,
                "text": _decode_span(tokenizer, response_ids, (start, end)),
                "information_text": tokenizer.decode(turn_ids[info_only], skip_special_tokens=True),
                "is_valid_search": is_search_turn,
                "is_answer_turn": is_answer_turn,
                "trainable_tokens": int(trainable.sum().item()),
                "raw_ig": float(output.non_tensor_batch["raw_ig_values"][i][turn_idx]),
                "turn_ig_reward": float(output.non_tensor_batch["turn_ig_rewards"][i][turn_idx]),
                "signed_ig": signed_ig,
                "ig_score": ig_score,
                "ig_credit_weight": ig_credit_weight,
                "ig_advantage": ig_advantage,
                "ig_reward": ig_credit_weight,
                "allocation_kind": allocation_kind,
                "allocation_component": allocation_component,
                "allocated_advantage": allocated_advantage,
                "deleted_target_mean_logprob": None if turn_idx >= len(deleted_scores) else float(deleted_scores[turn_idx]),
                "token_reward_sum": float(token_rewards[i, start:end].sum().item()),
                "trainable_token_reward_mean": float(trainable_rewards.mean().item()) if trainable_rewards.numel() else 0.0,
                "policy_mean_logprob": float(rollout_log_probs[i, start:end][response_attention[i, start:end]].mean().item()) if end > start else None,
                "context_without_turn": tokenizer.decode(deleted_context_ids, skip_special_tokens=True),
            })

        rm = batch.non_tensor_batch["reward_model"][i]
        records.append({
            "question_index": int(question_indices[i // n_rollouts]),
            "rollout": int(i % n_rollouts),
            "prompt": _jsonable(batch.non_tensor_batch.get("raw_prompt", [None] * len(batch))[i]),
            "ground_truth": _jsonable(rm["ground_truth"]),
            "pred_answer": _jsonable(output.non_tensor_batch["pred_answer"][i]),
            "response_text": _jsonable(output.non_tensor_batch["response_text"][i]),
            "final_f1": final_f1,
            "total_reward": total_reward,
            "sample_advantage": sample_advantage,
            "final_advantage": sample_advantage,
            "final_credit_weight": final_credit_weight,
            "ig_tau": _jsonable(batch.non_tensor_batch.get("ig_tau_values", [None] * len(batch))[i]),
            "ig_target_source": str(output.non_tensor_batch["ig_target_sources"][i]),
            "ig_enabled": bool(output.non_tensor_batch["ig_enabled"][i]),
            "full_target_mean_logprob": _jsonable(output.non_tensor_batch["full_target_log_probs"][i]),
            "trajectory_mean_logprob": float(rollout_log_probs[i, :valid_len].mean().item()) if valid_len else None,
            "full_context_before_target": tokenizer.decode(full_context_ids, skip_special_tokens=True),
            "turns": turns,
        })
    return records


def _mean(values):
    return float(statistics.mean(values)) if values else 0.0


def _median(values):
    return float(statistics.median(values)) if values else 0.0


def _summarize_records(records):
    turn_rows = [turn for record in records for turn in record["turns"]]
    search_rows = [turn for turn in turn_rows if turn["is_valid_search"]]
    raw_values = [float(turn["raw_ig"]) for turn in turn_rows]
    signed_values = [float(turn["signed_ig"]) for turn in turn_rows]
    ig_scores = [float(turn["ig_credit_weight"]) for turn in turn_rows]
    allocated_values = [float(turn["allocated_advantage"]) for turn in turn_rows]
    positive_raw = [x for x in raw_values if x > 0.0]
    negative_raw = [x for x in raw_values if x < 0.0]
    ig_score_sums = [float(sum(turn["ig_credit_weight"] for turn in record["turns"])) for record in records]
    weak_positive_rewards = [
        float(turn["ig_credit_weight"])
        for turn in turn_rows
        if 0.0 < float(turn["raw_ig"]) < 0.5 and float(turn["ig_credit_weight"]) > 0.0
    ]
    return {
        "num_records": len(records),
        "num_questions": len({record["question_index"] for record in records}),
        "mean_final_f1": _mean([float(record["final_f1"]) for record in records]),
        "mean_total_reward": _mean([float(record["total_reward"]) for record in records]),
        "sample_advantage_mean": _mean([float(record["sample_advantage"]) for record in records]),
        "sample_advantage_min": float(min(record["sample_advantage"] for record in records)) if records else 0.0,
        "sample_advantage_max": float(max(record["sample_advantage"] for record in records)) if records else 0.0,
        "nonzero_final_f1_ratio": _mean([float(record["final_f1"] > 0.0) for record in records]),
        "ig_enabled_ratio": _mean([float(record["ig_enabled"]) for record in records]),
        "num_turns": len(turn_rows),
        "num_search_turns": len(search_rows),
        "positive_raw_turns": len(positive_raw),
        "negative_raw_turns": len(negative_raw),
        "raw_ig_mean": _mean(raw_values),
        "raw_ig_median": _median(raw_values),
        "raw_ig_min": float(min(raw_values)) if raw_values else 0.0,
        "raw_ig_max": float(max(raw_values)) if raw_values else 0.0,
        "signed_ig_mean": _mean(signed_values),
        "signed_ig_min": float(min(signed_values)) if signed_values else 0.0,
        "signed_ig_max": float(max(signed_values)) if signed_values else 0.0,
        "ig_credit_weight_mean": _mean(ig_scores),
        "ig_credit_weight_max": float(max(ig_scores)) if ig_scores else 0.0,
        "ig_credit_weight_sum_mean": _mean(ig_score_sums),
        "ig_credit_weight_sum_median": _median(ig_score_sums),
        "ig_credit_weight_sum_max": float(max(ig_score_sums)) if ig_scores else 0.0,
        "allocated_advantage_mean": _mean(allocated_values),
        "allocated_advantage_min": float(min(allocated_values)) if allocated_values else 0.0,
        "allocated_advantage_max": float(max(allocated_values)) if allocated_values else 0.0,
        "weak_positive_reward_count": len(weak_positive_rewards),
        "weak_positive_reward_sum": float(sum(weak_positive_rewards)),
    }


def _write_report(records, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "diagnostics.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = _summarize_records(records)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    cards = []
    for record in records:
        turn_rows = []
        for turn in record["turns"]:
            width = min(100.0, abs(turn["allocated_advantage"]) * 100.0)
            color = "#22c55e" if turn["raw_ig"] > 0 else "#ef4444"
            turn_rows.append(f"""
              <div class="turn">
                <div><b>Turn {turn['turn']}</b> raw IG {turn['raw_ig']:.5f}, signed {turn['signed_ig']:.4f}, credit {turn['ig_credit_weight']:.4f}, token advantage {turn['allocated_advantage']:.4f}</div>
                <div class="bar"><span style="width:{width:.2f}%;background:{color}"></span></div>
                <div class="scores">kind {turn['allocation_kind']} | component {turn['allocation_component']:.4f} | token mean {turn['trainable_token_reward_mean']:.4f} | full {record['full_target_mean_logprob']} | delete {turn['deleted_target_mean_logprob']} | trainable tokens {turn['trainable_tokens']}</div>
                <pre>{html.escape(turn['text'])}</pre>
                <details><summary>Context after hard-delete</summary><pre>{html.escape(turn['context_without_turn'])}</pre></details>
              </div>""")
        cards.append(f"""
          <section>
            <h2>Question {record['question_index']} / rollout {record['rollout']}</h2>
            <div class="summary">F1 <b>{record['final_f1']:.3f}</b> | final advantage {record['sample_advantage']:.3f} | IG tau {float(record['ig_tau'] or 0.0):.5f} | target {html.escape(record['ig_target_source'])} | answer: {html.escape(str(record['pred_answer']))}</div>
            <details open><summary><b>Full generated response</b></summary><pre>{html.escape(str(record['response_text']))}</pre></details>
            <div class="scores">The turn blocks below show the token-limited text used by the IG calculation.</div>
            {''.join(turn_rows)}
          </section>""")

    report_path = output_dir / "report.html"
    report_path.write_text(f"""<!doctype html><meta charset="utf-8"><title>Counterfactual IG diagnostic</title>
<style>body{{font:14px system-ui;margin:24px;background:#f8fafc;color:#172033}}section{{background:white;padding:18px;margin:18px 0;border:1px solid #dbe3ef;border-radius:10px}}h2{{margin-top:0}}.summary,.scores{{color:#536177;margin:8px 0}}.turn{{border-top:1px solid #e5e7eb;padding-top:12px;margin-top:12px}}pre{{white-space:pre-wrap;background:#f3f5f8;padding:10px;border-radius:6px}}.bar{{height:9px;background:#e5e7eb;border-radius:9px;margin:7px 0;overflow:hidden}}.bar span{{height:100%;display:block}}</style>
<h1>Counterfactual IG diagnostic</h1>{''.join(cards)}""", encoding="utf-8")
    return jsonl_path, report_path, summary_path, summary


@hydra.main(config_path="config", config_name="counterfactual_ig_diagnostic", version_base=None)
def main(config):
    random.seed(int(config.diagnostic.seed))
    np.random.seed(int(config.diagnostic.seed))
    torch.manual_seed(int(config.diagnostic.seed))
    OmegaConf.resolve(config)

    if not ray.is_initialized():
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})

    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    dataset = RLHFDataset(
        parquet_files=config.diagnostic.data_file,
        tokenizer=tokenizer,
        prompt_key=config.data.prompt_key,
        max_prompt_length=config.data.max_prompt_length,
        filter_prompts=True,
        return_raw_chat=config.data.get("return_raw_chat", False),
        truncation="error",
    )
    selected = _select_indices(len(dataset), config.diagnostic)
    loader = DataLoader(Subset(dataset, selected), batch_size=len(selected), shuffle=False, collate_fn=collate_fn)
    batch = DataProto.from_single_dict(next(iter(loader)))
    n_rollouts = int(config.diagnostic.n_rollouts)
    batch = batch.repeat(repeat_times=n_rollouts, interleave=True)
    batch.non_tensor_batch["uid"] = np.repeat(np.array(selected, dtype=object), n_rollouts)

    actor_rollout, worker_container = _init_actor_rollout(config)
    gen_config = GenerationConfig(
        max_turns=config.max_turns,
        max_start_length=config.data.max_start_length,
        max_prompt_length=config.data.max_prompt_length,
        max_response_length=config.data.max_response_length,
        max_obs_length=config.data.max_obs_length,
        num_gpus=config.trainer.n_gpus_per_node * config.trainer.nnodes,
        no_think_rl=config.algorithm.no_think_rl,
        search_url=config.retriever.url,
        topk=config.retriever.topk,
        info_gain_type=getattr(config.algorithm, "info_gain_type", "log_prob_diff"),
        log_prob_micro_batch_size=config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
        log_prob_use_dynamic_bsz=config.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz,
        log_prob_max_token_len_per_gpu=config.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu,
        temperature=config.actor_rollout_ref.rollout.temperature,
        use_counterfactual_ig=True,
        ig_eps=config.algorithm.ig_eps,
        disable_old_gt_ig=True,
        lotapo_score_type=getattr(config.algorithm, "lotapo_score_type", "logprob"),
        lotapo_score_direction=getattr(config.algorithm, "lotapo_score_direction", "backward"),
    )
    manager = LLMGenerationManager(tokenizer, actor_rollout, gen_config)
    gen_batch = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
    gen_batch.meta_info = {
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "recompute_log_prob": False,
        "do_sample": bool(config.diagnostic.do_sample),
        "validate": False,
    }
    ground_truths = [rm["ground_truth"] for rm in batch.non_tensor_batch["reward_model"]]
    first_input_ids = gen_batch.batch["input_ids"][:, -gen_config.max_start_length:].clone().long()
    output = manager.run_llm_loop(gen_batch, first_input_ids, ground_truths)
    for key in output.batch.keys():
        output.batch[key] = output.batch[key].long()

    # Match the training path in RayPPOTrainer: compute_log_prob is executed by
    # the FSDP actor and requires these settings in DataProto.meta_info.
    output.meta_info.update({
        "micro_batch_size": config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
        "temperature": config.actor_rollout_ref.rollout.temperature,
        "max_token_len": config.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu,
        "use_dynamic_bsz": config.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz,
    })
    with torch.no_grad():
        log_prob_output = manager._compute_log_prob_with_gpu_padding(output)
    rollout_log_probs = log_prob_output.batch["old_log_probs"].detach().cpu()
    batch = batch.union(output)

    reward_manager = RewardManager(tokenizer=tokenizer, num_examine=0, algorithm_config=config.algorithm)
    token_rewards = reward_manager(batch)
    records = _records_from_batch(tokenizer, batch, output, token_rewards, rollout_log_probs, selected, n_rollouts)
    jsonl_path, report_path, summary_path, summary = _write_report(records, Path(config.diagnostic.output_dir).resolve())
    print(f"Wrote {len(records)} rollout diagnostics to {jsonl_path}")
    print(f"Wrote summary to {summary_path}: {json.dumps(summary, ensure_ascii=False)}")
    print(f"Open {report_path}")
    del worker_container
    ray.shutdown()


if __name__ == "__main__":
    main()
