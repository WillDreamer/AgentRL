"""
Borrowed from verl.trainer.main_ppo.py
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from recipe.webshop.trainer.agent_trainer import RayAgentTrainer

import ray
import hydra
import os, json, time, socket, uuid, atexit
from verl import DataProto
import torch
import numpy as np
from recipe.webshop.utils import register_resolvers
register_resolvers()
import sys
os.environ['VLLM_USE_V1'] = '1'


def _to_bool(x: str) -> bool:
    return str(x).lower() in {"1", "true", "t", "yes", "y", "on"}

class DummyRewardManager():
    """
    - 每过 ROLLOUT_SAVE_EVERY_STEPS 个 step 批量保存到“单独文件”
    - 文件名包含本批 step 区间：rollouts_s{start}_e{end}_rank{...}_pid{...}_{host}.jsonl
    - 若传入 compute_score，则用它算 reward；否则优先 rm_scores（均值），再回退 non_tensor_batch['reward']
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score

        # —— 步计数 + 缓冲 —— #
        self.step = 0
        self._buf = []
        self._buf_start_step = 1  # 当前缓冲的起始 step
        self.save_every_steps = int(os.getenv("ROLLOUT_SAVE_EVERY_STEPS", 5))  # “每过 N 步”
        env_flag = os.getenv("ROLLOUT_DUMP_ENABLE")
        self.save_rollout = _to_bool(env_flag) if env_flag is not None else False

        # —— 输出目录（默认当前工作目录/rollouts，Hydra 下就是当次 run 目录） —— #
        out_dir = os.environ.get("ROLLOUT_DUMP_DIR", os.path.join(os.getcwd(), "rollouts"))
        os.makedirs(out_dir, exist_ok=True)
        self._out_dir = out_dir

        # —— 进程标识 —— #
        self._host = socket.gethostname()
        self._rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
        self._pid  = os.getpid()

        print(f"[rollout-dump] enabled={self.save_rollout}, every={self.save_every_steps} steps")
        print(f"[rollout-dump] dir={self._out_dir}")

        # 退出兜底：写掉剩余缓冲
        atexit.register(self._flush, force=True)

    # === 当前分片文件名（按 step 分片，单独保存） ===
    def _current_file(self):
        return os.path.join(
            self._out_dir,
            f"rollouts_s{self._buf_start_step:06d}_e{self.step:06d}_"
            f"rank{self._rank}_pid{self._pid}_{self._host}.jsonl"
        )

    # === 缓冲写入 ===
    def _dump_one(self, rec: dict):
        if not self.save_rollout:
            return
        self._buf.append(rec)

    # === 批量落盘（生成独立文件），并滚动到下一个区间 ===
    def _flush(self, force: bool=False):
        if not self.save_rollout or not self._buf:
            # 无论写没写，都把下个区间的起点设成 step+1（防止命名错乱）
            self._buf_start_step = self.step + 1
            return
        path = self._current_file()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in self._buf))
            f.write("\n")
        self._buf.clear()
        # 下个分片从下一步开始
        self._buf_start_step = self.step + 1

    def __call__(self, data: "DataProto", return_dict=False):
        has_rm = "rm_scores" in data.batch

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        all_scores = []
        already_print_data_sources = {}

        for i in range(len(data)):
            item = data[i]

            # —— 1) 有效长度（左填充假设） ——
            seq_ids = item.batch["input_ids"]
            attn = item.batch['attention_mask']
            T = seq_ids.shape[-1]
            valid_seq_len = int(attn.sum().item())

            if "loss_mask" in item.batch:
                resp_len = int(item.batch["loss_mask"][i].sum().item())       # ★ 注意加 [i]
            elif "response_mask" in item.batch:
                resp_len = int(item.batch["response_mask"][i].sum().item())
            else:
                resp_len = int((item.batch["responses"][i] != 0).sum().item())

            resp_len = max(min(resp_len, valid_seq_len), 0)
            prompt_len = max(valid_seq_len - resp_len, 0)

            start_valid = T - valid_seq_len
            prompt_end  = T - resp_len
            valid_prompt_ids = seq_ids[start_valid:prompt_end]
            valid_resp_ids   = seq_ids[prompt_end:T]

            # —— 2) 解码 ——
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_resp_ids,  skip_special_tokens=True)
            sequences_str = self.tokenizer.decode(
                torch.cat((valid_prompt_ids, valid_resp_ids)),
                skip_special_tokens=True
            )

            # —— 3) 元信息 ——
            nt = item.non_tensor_batch or {}
            data_source  = nt.get("data_source", "default")
            ground_truth = nt.get("ground_truth")
            sample_index = nt.get("sample_index")
            rollout_n    = nt.get("rollout_n")

            # —— 4) 标量 reward（优先 rm_scores 的均值；也可改成 last/mean/sum 策略） ——
            if self.compute_score is not None:
                try:
                    score = float(self.compute_score(data_source, response_str, ground_truth, extra_info=nt))
                except TypeError:
                    score = float(self.compute_score(data_source, response_str, ground_truth))
            elif has_rm:
                score = float(data.batch['rm_scores'][i][-1].item())
            else:
                score = float(nt['reward']) if 'reward' in nt else 0.0

            if resp_len > 0:
                reward_tensor[i, resp_len - 1] = score
            all_scores.append(score)

            # —— 5) 进缓冲（到步数就生成“单独文件”） ——
            rec = {
                "ts": time.time(),
                "pid": os.getpid(),
                "prompt":   {"text": prompt_str,   "len_tokens": prompt_len},
                "response": {"text": response_str, "len_tokens": resp_len},
                "sequence": sequences_str,
                "ground_truth": ground_truth,
                "reward": score,
                "extra": {k: (str(v) if not isinstance(v, (str,int,float,bool,type(None))) else v)
                          for k, v in nt.items()},
            }
            self._dump_one(rec)

            # 少量样例打印
            if already_print_data_sources.get(data_source, 0) < self.num_examine:
                already_print_data_sources[data_source] = already_print_data_sources.get(data_source, 0) + 1
                print(sequences_str)

        print(f"[DEBUG] all_scores: {[f'{score:.4f}' for score in all_scores]}")


        # —— 步进：到阈值就写“独立文件” —— 
        self.step += 1
        if self.save_rollout and (self.step % self.save_every_steps == 0):
            self._flush()

        # —— 返回语义保持一致 —— 
        if has_rm:
            return {"reward_tensor": data.batch['rm_scores']} if return_dict else data.batch['rm_scores']
        else:
            return {"reward_tensor": reward_tensor} if return_dict else reward_tensor



# class DummyRewardManager():
#     """
#     Drop-in replacement:
#     - 全量将每条 rollout 落到 JSONL（每进程一个文件，避免写冲突）
#     - 若传入 compute_score，则用它算 reward；否则从 non_tensor_batch['reward'] 读取
#     """

#     def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
#         self.tokenizer = tokenizer
#         self.num_examine = num_examine
#         self.compute_score = compute_score

#         # 统一落盘目录：优先环境变量 ROLLOUT_DUMP_DIR，否则当前目录 ./rollouts
#         out_dir = os.environ.get("ROLLOUT_DUMP_DIR", "outputs/rollouts")
#         os.makedirs(out_dir, exist_ok=True)
#         env_flag = os.getenv("ROLLOUT_DUMP_ENABLE")
#         self.save_rollout = _to_bool(env_flag) if env_flag is not None else False

#         host = socket.gethostname()
#         rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
#         pid = os.getpid()
#         # 每个进程单独文件，减少并发写冲突
#         self._file = os.path.join(out_dir, f"rollouts_rank{rank}_pid{pid}_{host}.jsonl")


#     def _dump_one(self, rec: dict):
#         if not self.save_rollout:
#             return
#         # 逐行追加写入，简单稳妥；如需更高吞吐可考虑缓冲批量写
#         with open(self._file, "a", encoding="utf-8") as f:
#             f.write(json.dumps(rec, ensure_ascii=False) + "\n")

#     def __call__(self, data: "DataProto", return_dict=False):
#         # 不再提前 return，而是标记一下
#         has_rm = "rm_scores" in data.batch

#         reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
#         all_scores = []
#         already_print_data_sources = {}

#         for i in range(len(data)):
#             item = data[i]  # DataProtoItem

#             # ========== 1) 计算 prompt/response 的有效长度 ==========
#             # prompt_ids = item.batch['input_ids']
#             # prompt_len = prompt_ids.shape[-1]

#             seq_ids = item.batch["input_ids"] 
#             attn = item.batch['attention_mask']
#             T       = seq_ids.shape[-1]
#             valid_seq_len = int(attn.sum().item())

#             # ---- 计算 response 的有效长度 ----
#             if "loss_mask" in item.batch:               # 常见：loss_mask 就是 response mask
#                 resp_len = int(item.batch["loss_mask"].sum().item())
#             elif "response_mask" in item.batch:
#                 resp_len = int(item.batch["response_mask"].sum().item())
#             else:
#                 # 兜底：用 responses 的有效长度（若有右侧 padding 需根据实际pad_id处理）
#                 resp_len = int((item.batch["responses"] != 0).sum().item())

#             resp_len = max(min(resp_len, valid_seq_len), 0)
#             prompt_len = max(valid_seq_len - resp_len, 0)

#             # ---- 按“左填充”切片（VeRL 默认左填充，valid 在末尾）----
#             start_valid = T - valid_seq_len
#             prompt_end  = T - resp_len
#             valid_prompt_ids = seq_ids[start_valid:prompt_end]
#             valid_resp_ids   = seq_ids[prompt_end:T]

#             # ========== 2) 解码文本 ==========
#             prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
#             response_str = self.tokenizer.decode(valid_resp_ids,  skip_special_tokens=True)
#             sequences_str = self.tokenizer.decode(
#                 torch.cat((valid_prompt_ids, valid_resp_ids)),
#                 skip_special_tokens=True
#             )

#             # ========== 3) 收集元信息 ==========
#             nt = item.non_tensor_batch or {}
#             data_source  = nt.get("data_source", "default")
#             ground_truth = nt.get("ground_truth")
#             sample_index = nt.get("sample_index")
#             rollout_n    = nt.get("rollout_n")

#             # ========== 4) 计算/读取 reward ==========
#             if self.compute_score is not None:
#                 try:
#                     score = float(self.compute_score(data_source, response_str, ground_truth, extra_info=nt))
#                 except TypeError:
#                     score = float(self.compute_score(data_source, response_str, ground_truth))
#             elif has_rm:
#                 score = float(data.batch['rm_scores'][i].mean().item())
#             else:
#                 score = float(nt['reward']) if 'reward' in nt else 0.0

#             # 写回到 reward_tensor（最后一个有效 token 位置）
#             if resp_len > 0:
#                 reward_tensor[i, resp_len - 1] = score
#             all_scores.append(score)

#             # ========== 5) JSONL 落盘（每条样本一行） ==========
#             rec = {
#                 "ts": time.time(),
#                 "pid": os.getpid(),
#                 "prompt":   {"text": prompt_str,   "len_tokens": prompt_len},
#                 "response": {"text": response_str, "len_tokens": resp_len},
#                 "sequence": sequences_str,
#                 "ground_truth": ground_truth,
#                 "reward": score,
#                 "extra": {k: (str(v) if not isinstance(v, (str,int,float,bool,type(None))) else v)
#                         for k, v in nt.items()},
#             }
#             # rec = {
#             #     "ts": time.time(),
#             #     "uuid": str(uuid.uuid4()),
#             #     "host": socket.gethostname(),
#             #     "pid": os.getpid(),
#             #     "data_source": data_source,
#             #     "sample_index": sample_index,
#             #     "rollout_n": rollout_n,
#             #     "prompt":   {"text": prompt_str,   "len_tokens": prompt_len},
#             #     "response": {"text": response_str, "len_tokens": resp_len},
#             #     "sequence": sequences_str,
#             #     "ground_truth": ground_truth,
#             #     "reward": score,
#             #     "extra": {k: (str(v) if not isinstance(v, (str,int,float,bool,type(None))) else v)
#             #             for k, v in nt.items()},
#             # }
#             self._dump_one(rec)

#             # 可选：只打印少量样例
#             if data_source not in already_print_data_sources:
#                 already_print_data_sources[data_source] = 0
#             if already_print_data_sources[data_source] < self.num_examine:
#                 already_print_data_sources[data_source] += 1
#                 print(sequences_str)

#         print(f"[DEBUG] all_scores: {all_scores}")
#         # print(f"[DEBUG] all_scores shape: {np.array(all_scores).shape}")
#         # print(f"[DEBUG] all_scores mean: {np.mean(all_scores)}")
#         # print(f"[DEBUG] all_scores max: {np.max(all_scores)}")
#         # print(f"[DEBUG] all_scores min: {np.min(all_scores)}")
#         # print(f"[DEBUG] all_scores std: {np.std(all_scores)}")

#         # —— 关键：保持原语义的返回 —— 
#         if has_rm:
#             return {"reward_tensor": data.batch['rm_scores']} if return_dict else data.batch['rm_scores']
#         else:
#             return {"reward_tensor": reward_tensor} if return_dict else reward_tensor

# class DummyRewardManager():
#     """The reward manager.
#     """

#     def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
#         self.tokenizer = tokenizer
#         self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
#         self.compute_score = compute_score

#     def __call__(self, data: DataProto, return_dict=False):
#         """We will expand this function gradually based on the available datasets"""

#         # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
#         if 'rm_scores' in data.batch.keys():
#             if return_dict:
#                 return {
#                     "reward_tensor": data.batch['rm_scores'],
#                 }
#             else:
#                 return data.batch['rm_scores']

#         reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

#         all_scores = []

#         already_print_data_sources = {}

#         for i in range(len(data)):
#             data_item = data[i]  # DataProtoItem

#             prompt_ids = data_item.batch['prompts']

#             prompt_length = prompt_ids.shape[-1]

#             valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
#             valid_prompt_ids = prompt_ids[-valid_prompt_length:]

#             response_ids = data_item.batch['responses']
#             valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
#             valid_response_ids = response_ids[:valid_response_length]

#             # decode
#             sequences = torch.cat((valid_prompt_ids, valid_response_ids))
#             sequences_str = self.tokenizer.decode(sequences)

#             score = data_item.non_tensor_batch['reward']
#             score = float(score)
 
#             reward_tensor[i, valid_response_length - 1] = score
#             all_scores.append(score)

#             # Get data_source from data_item if available, otherwise use a default value
#             data_source = data_item.non_tensor_batch.get('data_source', 'default')
            
#             if data_source not in already_print_data_sources:
#                 already_print_data_sources[data_source] = 0

#             if already_print_data_sources[data_source] < self.num_examine:
#                 already_print_data_sources[data_source] += 1
#                 print(sequences_str)
        
#         print(f"[DEBUG] all_scores: {all_scores}")
#         print(f"[DEBUG] all_scores shape: {np.array(all_scores).shape}")
#         print(f"[DEBUG] all_scores mean: {np.mean(all_scores)}")
#         print(f"[DEBUG] all_scores max: {np.max(all_scores)}")
#         print(f"[DEBUG] all_scores min: {np.min(all_scores)}")
#         print(f"[DEBUG] all_scores std: {np.std(all_scores)}")

#         if return_dict:
#             return {
#                 "reward_tensor": reward_tensor,
#             }
#         else:
#             return reward_tensor

def get_custom_reward_fn(config):
    import importlib.util, os

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    if spec is None:
        raise RuntimeError(f"Failed to create module spec from '{file_path}'")
        
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}")

    function_name = reward_fn_config.get("name")
    if not function_name:
        raise ValueError("Function name not specified in custom_reward_function config")

    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")

    return getattr(module, function_name)



def add_dependency_and_validate_config(config):

    # validate config
    assert config.micro_batch_size_per_gpu * config.trainer.n_gpus_per_node <= config.actor_rollout_ref.actor.ppo_mini_batch_size, \
        f"micro_batch_size_per_gpu * n_gpus_per_node ({config.micro_batch_size_per_gpu * config.trainer.n_gpus_per_node}) must be less than or equal to ppo_mini_batch_size ({config.actor_rollout_ref.actor.ppo_mini_batch_size})"
    assert config.actor_rollout_ref.actor.ppo_mini_batch_size % (config.micro_batch_size_per_gpu * config.trainer.n_gpus_per_node) == 0, \
        f"ppo_mini_batch_size ({config.actor_rollout_ref.actor.ppo_mini_batch_size}) must be divisible by micro_batch_size_per_gpu * n_gpus_per_node ({config.micro_batch_size_per_gpu * config.trainer.n_gpus_per_node})"
    assert "qwen" in config.model_path.lower() or (not config.enable_response_mask), \
        "response mask is currently only supported for qwen models"
    assert len(str(config.system.CUDA_VISIBLE_DEVICES).split(',')) == config.trainer.n_gpus_per_node, \
        f"CUDA_VISIBLE_DEVICES ({config.system.CUDA_VISIBLE_DEVICES}) must have the same number of GPUs as n_gpus_per_node ({config.trainer.n_gpus_per_node})"
    assert (config.actor_rollout_ref.rollout.tensor_model_parallel_size == config.trainer.n_gpus_per_node) or (not config.actor_rollout_ref.rollout.tp_size_check), \
        f"actor_rollout_ref.rollout.tensor_model_parallel_size ({config.actor_rollout_ref.rollout.tensor_model_parallel_size}) is recommended to be the same as n_gpus_per_node ({config.trainer.n_gpus_per_node}) to maximize rollout speed. You can set actor_rollout_ref.rollout.tp_size_check=False to disable this check."
    assert config.es_manager.train.env_groups * config.es_manager.train.group_size * config.actor_rollout_ref.rollout.rollout_filter_ratio >= config.actor_rollout_ref.actor.ppo_mini_batch_size, \
        f"env_groups * group_size * rollout_filter_ratio ({config.es_manager.train.env_groups * config.es_manager.train.group_size * config.actor_rollout_ref.rollout.rollout_filter_ratio}) must be greater than or equal to ppo_mini_batch_size ({config.actor_rollout_ref.actor.ppo_mini_batch_size}). Note that effective rollouts for update is env_groups * group_size * rollout_filter_ratio."
    assert config.algorithm.bi_level_gae == False or config.algorithm.adv_estimator == "gae", "BI_LEVEL_GAE is enabled, so config.algorithm.adv_estimator should be set to gae"
    assert config.algorithm.bi_level_gae == False or (not config.agent_proxy.use_turn_scores), "BI_LEVEL_GAE is enabled, but currently use_turn_scores are not correctly supported, so config.agent_proxy.use_turn_scores should be set to False" # This will be added later. Currently turn-scores are not correctly supported yet.
    # assert config.algorithm.bi_level_gae == False or config.agent_proxy.use_turn_scores, "BI_LEVEL_GAE is enabled, so config.agent_proxy.use_turn_scores should be set to True" # This will be added later. Currently turn-scores are not correctly supported yet.

    # add dependency
    config.data.train_batch_size = config.es_manager.train.env_groups * config.es_manager.train.group_size


    return config


@hydra.main(version_base=None, config_path="config", config_name="base")
def main(config):
    config = add_dependency_and_validate_config(config)
    print(f"config: {config}")

    run_ppo(config)


def run_ppo(config) -> None:
    # TODO(linjunrong.ocss884): this ENV is left for resolving SGLang conflict with ray devices
    # isolation, will solve in the future
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.system.CUDA_VISIBLE_DEVICES)
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={
            'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN',
                'VLLM_LOGGING_LEVEL': 'WARN',
            }
        })
        # ray.init(
        #     address="auto",  # 或 None（本地）
        #     runtime_env={
        #         "env_vars": {
        #             "NCCL_SOCKET_IFNAME": "eth0",
        #             "NCCL_IB_DISABLE": "1",
        #             'TOKENIZERS_PARALLELISM': 'true',
        #             'NCCL_DEBUG': 'WARN',
        #             'VLLM_LOGGING_LEVEL': 'WARN',
        #         }
        #     }
        # )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}
    
    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker based on the actor strategy."""
        from verl.single_controller.ray import RayWorkerGroup

        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from recipe.webshop.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import Role

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)

        return actor_rollout_cls, ray_worker_group_cls

    def add_critic_worker(self, config):
        """Add critic worker to role mapping."""
        if config.critic.strategy in {"fsdp", "fsdp2"}:
            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable"]:
                from recipe.webshop.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import CriticWorker

                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

        elif config.critic.strategy == "megatron":
            from verl.workers.megatron_workers import CriticWorker

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import Role

        self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
    
    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager."""
        from verl.trainer.ppo.ray_trainer import Role

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        self.mapping[Role.ActorRollout] = global_pool_id
        self.mapping[Role.Critic] = global_pool_id
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)
        return resource_pool_manager

    def add_reward_model_worker(self, config):
        """Add reward model worker if enabled."""
        from verl.trainer.ppo.ray_trainer import Role

        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from recipe.webshop.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            self.role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            self.mapping[Role.RewardModel] = "global_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls):
        """Add reference policy worker if KL loss or KL reward is used."""
        from verl.trainer.ppo.ray_trainer import Role

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
            self.mapping[Role.RefPolicy] = "global_pool"

    def run(self, config):
        from verl.utils.fs import copy_to_local
        # print initial config
        from pprint import pprint

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        from verl.utils import hf_tokenizer, hf_processor
        tokenizer = hf_tokenizer(local_path)
        processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)

        # We should adopt a multi-source reward function here:
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # finally, we combine all the rewards together
        # The reward type depends on the tag of the data
        self.add_reward_model_worker(config)

        # Add a reference policy worker if KL loss or KL reward is used.
        self.add_ref_policy_worker(config, actor_rollout_cls)

        from verl.utils.config import validate_config
        from verl.trainer.ppo.utils import need_critic, need_reference_policy
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        print("using dummy reward manager")
        reward_manager_cls = DummyRewardManager

        compute_score = get_custom_reward_fn(config)
        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, compute_score=compute_score)
        # Note that we always use function-based RM for validation
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=compute_score)

        resource_pool_manager = self.init_resource_pool_mgr(config)

        trainer = RayAgentTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn
        )
        trainer.init_workers()
        trainer.init_agent_proxy()
        trainer.fit()


if __name__ == '__main__':
    main()
