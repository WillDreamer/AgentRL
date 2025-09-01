from .ctx_manager_bedrock import ContextManager_Bedrock
from .es_manager_bedrock import EnvStateManager_Bedrock
import torch
from verl import DataProto
from typing import List, Dict
import torch
# import time

class LLMAgentProxy_Bedrock:
	"""
	The proxy means the llm agent is trying to generate some rollout **at this time**, **at this model state**, **at this env state from the env config**
	"""
	def __init__(self, config, model):
		self.config = config
		self.model = model
		self.train_ctx_manager = ContextManager_Bedrock(config, mode="train")
		self.train_es_manager = EnvStateManager_Bedrock(config, mode="train")

	def rollout(self, val=False):
		es_manager = self.train_es_manager
		ctx_manager = self.train_ctx_manager
		env_outputs = es_manager.reset()

		response_mask = torch.zeros(len(env_outputs), self.config.agent_proxy.max_turn)

		for i in range(self.config.agent_proxy.max_turn):
			lm_inputs = ctx_manager.get_lm_inputs(env_outputs, prepare_for_update=False)
			
			lm_outputs = {}
			for idx, x in enumerate(lm_inputs):
				if response_mask[idx].sum() > 0:
					lm_outputs[x] = None
					continue
				response = self.model.respond(
					[lm_inputs[x][-1]],
					max_tokens=512,
					max_context_size=15000
				)
				lm_outputs[x] = response
				if response is not None:
					if 'click[buy now]' in response:
						response_mask[idx, i] = 1

			env_inputs: List[Dict] = ctx_manager.get_env_inputs(lm_outputs)
			env_outputs: List[Dict] = es_manager.step(env_inputs)
			if len(env_outputs) == 0: # all finished
				break
			
		rollout_states = es_manager.get_rollout_states() 
		rollouts, metrics = ctx_manager.formulate_rollouts(rollout_states)
		
		return rollouts, metrics
