
from recipe.webshop.llm_agent.ctx_manager_old import ContextManager
from recipe.webshop.llm_agent.es_manager_old import EnvStateManager
from vllm import LLM, SamplingParams
from verl.single_controller.ray.base import RayWorkerGroup
from transformers import AutoTokenizer, AutoModelForCausalLM
from verl import DataProto
import hydra
import os
from typing import List, Dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from recipe.webshop.llm_agent.base_llm import ConcurrentLLM
from recipe.webshop.llm_agent.agent_proxy_old import ApiCallingWrapperWg, VllmWrapperWg, LLMAgentProxy

@hydra.main(version_base=None, config_path="./config", config_name="base_eval")
def main(config):
	# detect config name from python -m webshop.llm_agent.agent_proxy --config_name frozen_lake
	os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
	os.environ["CUDA_VISIBLE_DEVICES"] = str(config.system.CUDA_VISIBLE_DEVICES)
	tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
	actor_wg = VllmWrapperWg(config, tokenizer)
	proxy = LLMAgentProxy(config, actor_wg, tokenizer)
	import time
	start_time = time.time()
	rollouts = proxy.rollout(DataProto(batch=None, non_tensor_batch=None, meta_info={'eos_token_id': 151645, 'pad_token_id': 151643, 'recompute_log_prob': False, 'do_sample':config.actor_rollout_ref.rollout.do_sample, 'validate': True}), val=True)

	end_time = time.time()
	print(f'rollout time: {end_time - start_time} seconds')
	breakpoint()
	# print rollout rewards from the rm_scores
	rm_scores = rollouts.batch["rm_scores"].sum(-1)
	mask = rm_scores > 0.99
	rollouts_saved = rollouts.non_tensor_batch["messages_list"][mask]

	# save rollouts_saved to json
	with open('rollouts_saved.json', 'w') as f:
		json.dump(rollouts_saved, f)


	

if __name__ == "__main__":
	main()
