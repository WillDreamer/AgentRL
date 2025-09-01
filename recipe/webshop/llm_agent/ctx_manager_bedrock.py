"""
This is the context manager for the LLM agent.
author: Kangrui Wang, Zihan Wang
date: 2025-03-30
"""
from itertools import zip_longest

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import re
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from transformers import AutoTokenizer
import hydra
from recipe.webshop.utils import register_resolvers
from recipe.webshop.env import REGISTERED_ENV_CONFIGS
from tensordict import TensorDict

from dataclasses import asdict
register_resolvers()

def get_special_tokens(tokenizer: AutoTokenizer):
    if "qwen" in tokenizer.name_or_path.lower():
        special_token = tokenizer.encode("<|im_start|>")[0]
        reward_token = tokenizer.encode("<|im_end|>")[0]
    elif "llama-3" in tokenizer.name_or_path.lower():
        special_token = 128006
        reward_token = 128009
    else:
        raise ValueError(f"Unsupported model: {tokenizer.name_or_path}")
    return special_token, reward_token

def get_masks_and_scores(input_ids: torch.Tensor, tokenizer: AutoTokenizer, all_scores: List[List[float]] = None, use_turn_scores: bool = False, enable_response_mask: bool = False):
    """
    input_ids: shape (bsz, seq_len)
    Get loss mask that only learns between <|im_start|>assistant and <|im_end|>. Currently only supports qwen.
    NOTE: important! This assumes that the input_ids starts with system and then user & assistant in alternative ways
    """
    special_token, reward_token = get_special_tokens(tokenizer)
    
    turn_starts = torch.where(input_ids == special_token, 1, 0)
    turn_indicators = torch.cumsum(turn_starts, dim=-1)
    if enable_response_mask:
        loss_mask = (turn_indicators % 2 == 1) & (turn_indicators > 1) # only learns all assistant turns
    else:
        loss_mask = (turn_indicators > 1) # learns everything after system prompt
    response_mask = (turn_indicators % 2 == 1) & (turn_indicators > 1)
    
    score_tensor = torch.zeros_like(input_ids, dtype=torch.float32)
    if use_turn_scores:
        for idx, scores in enumerate(zip_longest(*all_scores, fillvalue=0)):
            scores = torch.tensor(scores, dtype=torch.float32)
            turn_indicator = idx * 2 + 3 # 0: pad. 1: system. 2+2n: user. 3+2n: assistant
            reward_position = (input_ids == reward_token) & (turn_indicators == turn_indicator)
            # Set the last token of the rows where all positions are False to True
            reward_position[~reward_position.any(dim=-1), -1] = True
            score_tensor[reward_position] = scores
        if "qwen" in tokenizer.name_or_path.lower():
            # for Qwen, there is a "\n" between special token and reward token, so we shift this to make sure reward is assigned to the last token of a turn
            score_tensor = score_tensor.roll(shifts=1, dims=-1)
    else:
        scores = [sum(i) for i in all_scores]
        score_tensor[:, -1] = torch.tensor(scores, dtype=torch.float32)
    score_tensor = score_tensor[:, 1:] # remove the first token
    loss_mask = loss_mask[:, :-1] # remove the last token
    response_mask = response_mask[:, :-1] # remove the last token

    return score_tensor, loss_mask, response_mask



class ContextManager_Bedrock:
    """
    Manages the context for LLM interactions with environments.
    Translates between environment outputs and LLM inputs, and vice versa.
    """

    def __init__(self, 
                 config,
                 processor = None,
                 mode: str = "train",
                 ):
        """
        Initialize the ContextManager.
        Processor is used to process the image data.
        """
        self.config = config
        self.processor = processor
        self.action_sep = self.config.agent_proxy.action_sep
        self.special_token_list = ["<think>", "</think>", "<answer>", "</answer>", "<|im_start|>", "<|im_end|>"]

        self.es_cfg = self.config.es_manager[mode]
        self.env_nums = {
                env_tag: n_group * self.es_cfg.group_size
                for n_group, env_tag in zip(self.es_cfg.env_configs.n_groups, self.es_cfg.env_configs.tags)
        }
        self._init_prefix_lookup()
    
    def _check_env_installed(self, env_type: str):
        if env_type not in REGISTERED_ENV_CONFIGS:
            raise ValueError(f"Environment {env_type} is not installed. Please install it using the scripts/setup_{env_type}.sh script.")

    def _init_prefix_lookup(self):
        prefix_lookup = {}
        prefixes = {}
        env_config_lookup = {}
        env_config = {}
        for env_tag, env_config in self.config.custom_envs.items():
            if env_tag not in self.es_cfg.env_configs.tags:
                continue

            self._check_env_installed(env_config.env_type)
            env_config_new = asdict(REGISTERED_ENV_CONFIGS[env_config.env_type]())
            for k,v in env_config.items():
                env_config_new[k] = v
            env_instruction = env_config_new.get("env_instruction", "")
            if env_config_new.get("grid_vocab", False):
                grid_vocab_str = "\nThe meaning of each symbol in the state is:\n" + ", ".join([f"{k}: {v}" for k, v in env_config_new["grid_vocab"].items()])
                env_instruction += grid_vocab_str
            if env_config_new.get("action_lookup", False):
                action_lookup_str = "\nYour available actions are:\n" + ", ".join([f"{v}" for k, v in env_config_new["action_lookup"].items()])
                action_lookup_str += f"\nYou can make up to {env_config_new['max_actions_per_traj']} actions, separated by the action separator \" " + self.action_sep + " \"\n"
                env_instruction += action_lookup_str
            prefixes[env_tag] = env_instruction
            env_config_lookup[env_tag] = {'max_tokens': env_config.get("max_tokens", self.config.actor_rollout_ref.rollout.response_length)}

        tags = self.es_cfg.env_configs.tags
        n_groups = self.es_cfg.env_configs.n_groups
        group_size = self.es_cfg.group_size

        cur_group = 0
        for env_tag, n_group in zip(tags, n_groups):
            env_instruction = prefixes[env_tag]
            start_idx = cur_group * group_size
            end_idx = (cur_group + n_group) * group_size
            for i in range(start_idx, end_idx):
                prefix_lookup[i] = env_instruction
                env_config_lookup[i] = env_config_lookup[env_tag]
            cur_group += n_group
            
        self.prefix_lookup = prefix_lookup
        self.env_config_lookup = env_config_lookup

    def _parse_response(self, response: str) -> List:
        pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>' if self.config.agent_proxy.enable_think else r'<answer>(.*?)</answer>'
        match = re.search(pattern, response, re.DOTALL)
        if not match:
            # think_content, action_content, actions = "", "", [] # do not remove this kind of invalid string
            llm_response, actions = response, []
        else:
            if self.config.agent_proxy.enable_think:
                think_content, action_content = match.group(1), match.group(2)
            else:
                think_content, action_content = "", match.group(1)

            for special_token in self.special_token_list:
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()
            
            actions = [action.strip() for action in action_content.split(self.action_sep) if action.strip()]
            max_actions = self.config.agent_proxy.max_actions_per_turn

            if len(actions) > max_actions:
                actions = actions[:max_actions] #Only the first MAX_ACTIONS actions are kept in the rollout.
                action_content = (" " + self.action_sep + " ").join(actions)

            llm_response = f"<think>{think_content}</think><answer>{action_content}</answer>" if self.config.agent_proxy.enable_think else f"<answer>{action_content}</answer>"
        return llm_response, actions
        
    def _normalize_score_tensor(self, score_tensor: torch.Tensor, env_outputs: List[Dict]) -> torch.Tensor:
        """
        Normalize the score tensor to be between 0 and 1.
        NOTE: only support score at the last token for now
        """
        assert self.config.agent_proxy.use_turn_scores == False, "Reward normalization is not supported for use_turn_scores == True"
        
        rn_cfg = self.config.agent_proxy.reward_normalization
        grouping, method = rn_cfg.grouping, rn_cfg.method
        if grouping == "state":
            group_tags = [env_output["group_id"] for env_output in env_outputs]
        elif grouping == "inductive":
            group_tags = [env_output["tag"] for env_output in env_outputs]
        elif grouping == "batch":
            group_tags = [1] * len(env_outputs)
        else:
            raise ValueError(f"Invalid grouping: {grouping}")


        if method == "mean_std":
            norm_func = lambda x: (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6) if x.std(dim=-1, keepdim=True).abs().max() > 1e-6 else torch.zeros_like(x) # stable to bf16 than x.std()
        elif method == "mean":
            norm_func = lambda x: (x - x.mean(dim=-1, keepdim=True))
        elif method == "asym_clip":
            norm_func = lambda x: ((x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6) if x.std(dim=-1, keepdim=True).abs().max() > 1e-6 else torch.zeros_like(x)).clamp(min=-1, max=3)
        elif method == "identity":
            norm_func = lambda x: x
        else:
            raise ValueError(f"Invalid normalization method: {method}")

        # apply groupwise normalization
        group2index = {}
        for i, env_tag in enumerate(group_tags):
            if env_tag not in group2index:
                group2index[env_tag] = []
            group2index[env_tag].append(i)
        group2index = {k: torch.tensor(v) for k, v in group2index.items()}

        
        # apply penalty pre-normalization
        acc_scores = score_tensor[:, -1]
        normalized_acc_scores = acc_scores.clone()
        penalty = torch.tensor([env_output.get("penalty", 0) for env_output in env_outputs], dtype=torch.float32)
        normalized_acc_scores = normalized_acc_scores + penalty

        if len(group2index) < acc_scores.shape[0]: # the group size > 1
            for group, index in group2index.items():
                normalized_acc_scores[index] = norm_func(normalized_acc_scores[index])

        score_tensor[:, -1] = normalized_acc_scores

        return score_tensor
    
    def get_lm_inputs(self, env_outputs: List[Dict], prepare_for_update: bool) -> DataProto:
        """
        env_outputs - please see below example
        [
            {"env_id": 1, "history": [{"state": "###\n#x_#", "llm_response": "Response 1", "reward": 0.5}, {"state": "###\n#x_#"}]},
            {"env_id": 2, "history": [{"state": "###\n#x_#"}]},
            ...
        ]
        prefix_lookup - from env_id to initial prompt
        """

        messages_list = {} # for api calling
        
        for env_output in env_outputs:
            env_id = env_output["env_id"]
            if 'state' in env_output['history'][-1] and prepare_for_update:
                env_output['history'] = env_output['history'][:-1] # when prepare for update, we do not add the state from the n+1 turn to the trajectory
            
            max_k = getattr(self.config.agent_proxy, "max_context_window", None)
            if max_k is not None and isinstance(max_k, int) and max_k > 0:
                env_output['history'] = env_output['history'][-max_k:]
            
            messages = [
                {"role": "system", "content": f"You're a helpful assistant. "}, 
                {"role": "user", "content": self.prefix_lookup[env_output["env_id"]]}
            ]

            for idx, content in enumerate(env_output["history"]):
                messages[-1]["content"] += f"\nTurn {idx + 1}:\n"
                if "state" in content:
                    FORMAT_PROMPT = "<think> [Your thoughts] </think> <answer> [your answer] </answer>" if self.config.agent_proxy.enable_think else "<answer> [your answer] </answer>"
                    LENGTH_PROMPT = f"Max response length: {self.env_config_lookup[env_output['env_id']]['max_tokens']} words (tokens)."
                    messages[-1]["content"] += f"State:\n{content['state']}\nYou have {content['actions_left']} actions left. Always output: {FORMAT_PROMPT} with no extra text. Strictly follow this format. {LENGTH_PROMPT}\n"
                if "llm_response" in content:
                    messages.append({"role": "assistant", "content": content["llm_response"]})
                if "reward" in content and not (prepare_for_update and idx == len(env_output["history"]) - 1):
                    # when prepare for update, we do not add the reward from the n+1 turn to the trajectory
                    messages.append({"role": "user", "content": f"Reward:\n{content['reward']}\n"})
                    
            # NOTE: this assertion is important for loss mask computation        
            assert all(msg["role"] == "assistant" for msg in messages[2::2])
            messages_list[env_id] = messages

        if prepare_for_update:
            metrics = {}
            for env_output in env_outputs:
                for key, value in env_output["metrics"].items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
            return messages_list, metrics
        else:
            return messages_list

    def get_env_inputs(self, lm_outputs) -> List[Dict]:
        
        env_inputs = []
        for env_id in lm_outputs:
            response = lm_outputs[env_id]
            if response is not None:
                llm_response, actions = self._parse_response(response)
            else:
                llm_response, actions = "", []
            env_inputs.append({
                "env_id": env_id,
                "llm_raw_response": response,
                "llm_response": llm_response,
                "actions": actions,
            })
        return env_inputs

    def formulate_rollouts(self, env_outputs: List[Dict]) -> DataProto:
        llm_inputs, metrics = self.get_lm_inputs(env_outputs, prepare_for_update=True)
        return llm_inputs, metrics


    
