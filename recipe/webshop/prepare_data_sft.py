
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
from recipe.webshop.llm_agent.agent_proxy_bedrock import LLMAgentProxy_Bedrock
from api.bedrock import get_model

@hydra.main(version_base=None, config_path="./config", config_name="base_bedrock")
def main(config):
	# detect config name from python -m webshop.llm_agent.agent_proxy --config_name frozen_lake
	os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
	os.environ["CUDA_VISIBLE_DEVICES"] = str(config.system.CUDA_VISIBLE_DEVICES)
	tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)

	model_name = 'anthropic.claude-3-5-sonnet-20240620-v1:0'
	region = 'us-east-1'
	model = get_model(model_name, None, region)
	proxy = LLMAgentProxy_Bedrock(config, model)
	import time
	start_time = time.time()
	rollouts, metrics = proxy.rollout(val=False)
	breakpoint()
	end_time = time.time()
	print(f'rollout time: {end_time - start_time} seconds')
	# print rollout rewards from the rm_scores
	print(f'metrics:')
	for k, v in metrics.items():
		print(f'{k}: {v}')

if __name__ == "__main__":
	main()
