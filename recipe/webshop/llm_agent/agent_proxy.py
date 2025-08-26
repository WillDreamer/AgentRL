from .ctx_manager import ContextManager
from .es_manager import EnvStateManager
from vllm import LLM, SamplingParams
from verl.single_controller.ray.base import RayWorkerGroup
from transformers import AutoTokenizer, AutoModelForCausalLM
from verl import DataProto
import hydra
import os
from typing import List, Dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from .base_llm import ConcurrentLLM
import torch
from api.bedrock import get_model
# import time


class VllmWrapperWg: # Thi is a developing class for eval and test
	def __init__(self, config, tokenizer):
		self.config = config
		self.tokenizer = tokenizer
		model_name = config.actor_rollout_ref.model.path
		ro_config = config.actor_rollout_ref.rollout
		self.llm = LLM(
			model_name,
            enable_sleep_mode=True,
            tensor_parallel_size=ro_config.tensor_model_parallel_size,
            dtype=ro_config.dtype,
            enforce_eager=ro_config.enforce_eager,
            gpu_memory_utilization=ro_config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            # disable_mm_preprocessor_cache=True,
            skip_tokenizer_init=False,
            max_model_len=ro_config.max_model_len,
            disable_log_stats=ro_config.disable_log_stats,
            max_num_batched_tokens=ro_config.max_num_batched_tokens,
            enable_chunked_prefill=ro_config.enable_chunked_prefill,
            enable_prefix_caching=True,
			trust_remote_code=True,
		)
		print("LLM initialized")
		self.sampling_params = SamplingParams(
			max_tokens=ro_config.response_length,
			temperature=ro_config.val_kwargs.temperature,
			top_p=ro_config.val_kwargs.top_p,
			top_k=ro_config.val_kwargs.top_k,
			# min_p=0.1,
		)

	def generate_sequences(self, lm_inputs: DataProto):
		"""
		Convert the input ids to text, and then generate the sequences. Finally create a dataproto. 
		This aligns with the verl Worker Group interface.
		"""
		# NOTE: free_cache_engine is not used in the vllm wrapper. Only used in the verl vllm.
		# cache_action = lm_inputs.meta_info.get('cache_action', None)

		input_ids = lm_inputs.batch['input_ids']
		input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
		input_texts = [i.replace("<|endoftext|>", "") for i in input_texts]

		outputs = self.llm.generate(input_texts, sampling_params=self.sampling_params)
		texts = [output.outputs[0].text for output in outputs] 
		lm_outputs = DataProto()
		lm_outputs.non_tensor_batch = {
			'response_texts': texts,
			'env_ids': lm_inputs.non_tensor_batch['env_ids'],
			'group_ids': lm_inputs.non_tensor_batch['group_ids']
		} # this is a bit hard-coded to bypass the __init__ check in DataProto
		lm_outputs.meta_info = lm_inputs.meta_info

		return lm_outputs
	
class ApiCallingWrapperWg:
    """Wrapper class for API-based LLM calls that fits into the VERL framework"""
    
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        model_info = config.model_info[config.model_config.model_name]
        self.llm_kwargs = model_info.generation_kwargs
        
        
        self.llm = ConcurrentLLM(
			provider=model_info.provider_name,
            model_name=model_info.model_name,
            max_concurrency=config.model_config.max_concurrency
        )
        
        print(f'API-based LLM ({model_info.provider_name} - {model_info.model_name}) initialized')


    def generate_sequences(self, lm_inputs: DataProto) -> DataProto:
        """
        Convert the input ids to text, make API calls to generate responses, 
        and create a DataProto with the results.
        """

        messages_list = lm_inputs.non_tensor_batch['messages_list'].tolist()
        results, failed_messages = self.llm.run_batch(
            messages_list=messages_list,
            **self.llm_kwargs
        )
        assert not failed_messages, f"Failed to generate responses for the following messages: {failed_messages}"

        texts = [result["response"] for result in results]
        print(f'[DEBUG] texts: {texts}')
        lm_outputs = DataProto()
        lm_outputs.non_tensor_batch = {
			'response_texts': texts,
			'env_ids': lm_inputs.non_tensor_batch['env_ids'],
			'group_ids': lm_inputs.non_tensor_batch['group_ids']
		} # this is a bit hard-coded to bypass the __init__ check in DataProto
        lm_outputs.meta_info = lm_inputs.meta_info
        
        return lm_outputs

class LLMAgentProxy:
	"""
	The proxy means the llm agent is trying to generate some rollout **at this time**, **at this model state**, **at this env state from the env config**
	"""
	def __init__(self, config, actor_rollout_wg, tokenizer):
		self.config = config
		self.train_ctx_manager = ContextManager(config, tokenizer, mode="train")
		self.train_es_manager = EnvStateManager(config, mode="train")
		self.val_ctx_manager = ContextManager(config, tokenizer, mode="val")
		self.val_es_manager = EnvStateManager(config, mode="val")
		self.actor_wg = actor_rollout_wg
		self.tokenizer = tokenizer

	def generate_sequences(self, lm_inputs: DataProto, val=False):
		# TODO: add kv cache both for the vllm wrapper here and for verl vllm.
		if isinstance(self.actor_wg, RayWorkerGroup):
			padded_lm_inputs, pad_size = pad_dataproto_to_divisor(lm_inputs, self.actor_wg.world_size)
			
			off_policy = True

			## off policy rollout
			if off_policy and not val:
				padded_lm_outputs = self.actor_wg.generate_sequences(padded_lm_inputs)

				B = padded_lm_outputs.batch.batch_size.numel()
				padded_lm_outputs.batch.set("activate_off", torch.zeros(B, dtype=torch.bool))

				offp_mask = padded_lm_inputs.batch["offp_mask"]==0
				td_off = padded_lm_inputs.batch[~offp_mask]
				non_off = {k: v[~offp_mask.numpy()] for k, v in padded_lm_inputs.non_tensor_batch.items()}
				data_off = DataProto(batch=td_off, non_tensor_batch=non_off)
				offp_model = get_model(model_name="anthropic.claude-3-5-sonnet-20240620-v1:0", openai_key=None, region="us-east-1")
				data_off_texts = self.tokenizer.batch_decode(data_off.batch['input_ids'],skip_special_tokens=True)
				responses_off = []
				breakpoint()
				for i in range(len(data_off_texts)):
					response = offp_model.respond(
								[
									{"role": "user", "content": data_off_texts[i]}
								],
								max_tokens=self.config.actor_rollout_ref.rollout.response_length,
								max_context_size=self.config.actor_rollout_ref.rollout.max_model_len
							)
					if response is not None:
						idx = torch.where(~offp_mask)[0][i].item()
						decode_text = self.tokenizer.encode(response)
						padded_lm_outputs[idx].batch['responses'] = decode_text +[self.tokenizer.pad_token_id for i in range(self.config.actor_rollout_ref.rollout.response_length - len(decode_text)) ]
						padded_lm_outputs[idx].batch['input_ids'][-self.config.actor_rollout_ref.rollout.response_length:] = padded_lm_outputs[idx].batch['responses']
						padded_lm_outputs[idx].batch['attention_mask'][-self.config.actor_rollout_ref.rollout.response_length:][:len(decode_text)]=1
						padded_lm_outputs[idx].batch['attention_mask'][-self.config.actor_rollout_ref.rollout.response_length:][len(decode_text):]=0
						padded_lm_outputs[idx].batch['activate_off'][idx] = True

					responses_off.append(response)
				breakpoint()
				
			else:
				padded_lm_outputs = self.actor_wg.generate_sequences(padded_lm_inputs)

			# idx_ = 2
			# tokens = padded_lm_outputs.batch['responses'][idx_][padded_lm_outputs.batch['responses'][idx_] != 151643]
			# decoded_tokens = [self.tokenizer.decode([tid], skip_special_tokens=True) for tid in tokens]
			# prob = padded_lm_outputs.batch['rollout_log_probs'][idx_][padded_lm_outputs.batch['rollout_log_probs'][idx_] != -1]
			lm_outputs = unpad_dataproto(padded_lm_outputs, pad_size=pad_size)
			lm_outputs.meta_info = lm_inputs.meta_info
			lm_outputs.non_tensor_batch = lm_inputs.non_tensor_batch
		elif isinstance(self.actor_wg, VllmWrapperWg) or isinstance(self.actor_wg, ApiCallingWrapperWg):
			lm_outputs = self.actor_wg.generate_sequences(lm_inputs)
		else:
			raise ValueError(f"Unsupported actor worker type: {type(self.actor_wg)}")

		return lm_outputs

	def _handle_void_actions(self, lm_inputs: DataProto, lm_outputs: DataProto, max_retries: int = 1) -> DataProto:
		"""
		Check for void actions (responses without <answer> tags) and regenerate them.
		
		Args:
			lm_inputs: Original inputs to the LLM
			lm_outputs: Outputs from the LLM
			max_retries: Maximum number of regeneration attempts
			
		Returns:
			Updated lm_outputs with regenerated responses for void actions
		"""
		current_outputs = lm_outputs
		
		for retry in range(max_retries):
			# Decode responses to check for <answer> tags
			if current_outputs.batch is not None and 'responses' in current_outputs.batch:
				# For tensor-based responses (RayWorkerGroup)
				decoded_responses = self.tokenizer.batch_decode(current_outputs.batch['responses'], skip_special_tokens=True)
			elif current_outputs.non_tensor_batch is not None and 'response_texts' in current_outputs.non_tensor_batch:
				# For text-based responses (VllmWrapperWg, ApiCallingWrapperWg)
				decoded_responses = current_outputs.non_tensor_batch['response_texts']
			else:
				# No responses to check
				return current_outputs
			
			# Create void mask: True for samples that need regeneration
			void_mask = [not ('<answer>' in r and '</answer>' in r) for r in decoded_responses]
			
			# If no void actions, return current outputs
			if not any(void_mask):
				return current_outputs
			
			print(f"Retry {retry + 1}: Found {sum(void_mask)} void actions out of {len(void_mask)} samples. Regenerating...")
			
			# Extract inputs for void samples
			void_indices = [i for i, is_void in enumerate(void_mask) if is_void]
			
			# Create new inputs for void samples only
			void_lm_inputs = self._extract_void_inputs(lm_inputs, void_indices)
			
			# Regenerate responses for void samples
			lm_outputs_new = self.generate_sequences(void_lm_inputs)
			
			# Update current outputs with new responses
			current_outputs = self._update_outputs_with_regenerated(current_outputs, lm_outputs_new, void_indices)
		
		# If we still have void actions after max retries, log a warning but continue
		final_decoded_responses = self.tokenizer.batch_decode(current_outputs.batch['responses'], skip_special_tokens=True) \
			if current_outputs.batch is not None and 'responses' in current_outputs.batch \
			else current_outputs.non_tensor_batch.get('response_texts', [])
		
		final_void_count = sum(1 for r in final_decoded_responses if not ('<answer>' in r and '</answer>' in r))
		if final_void_count > 0:
			print(f"Warning: After {max_retries} retries, still have {final_void_count} void actions. Proceeding anyway.")
		
		return current_outputs
	
	def _extract_void_inputs(self, lm_inputs: DataProto, void_indices: List[int]) -> DataProto:
		"""Extract inputs for void samples to regenerate them."""
		# Use DataProto's built-in indexing which properly handles TensorDict
		void_indices_tensor = torch.tensor(void_indices)
		void_lm_inputs = lm_inputs[void_indices_tensor]
		return void_lm_inputs
	
	def _update_outputs_with_regenerated(self, original_outputs: DataProto, new_outputs: DataProto, void_indices: List[int]) -> DataProto:
		"""Update original outputs with regenerated responses for void samples."""
		# Create a copy of the original outputs
		updated_outputs = DataProto()
		updated_outputs.meta_info = original_outputs.meta_info.copy()
		
		# Handle tensor batch data
		if original_outputs.batch is not None:
			# Clone the original batch data structure
			if hasattr(original_outputs.batch, 'clone'):
				updated_outputs.batch = original_outputs.batch.clone()
			else:
				# Fallback for dictionary-like batch data
				updated_outputs.batch = {}
				for key, tensor in original_outputs.batch.items():
					if isinstance(tensor, torch.Tensor):
						updated_outputs.batch[key] = tensor.clone()
					else:
						updated_outputs.batch[key] = tensor.copy() if hasattr(tensor, 'copy') else list(tensor)
			
			# Update void indices with new data
			if new_outputs.batch is not None:
				for key in updated_outputs.batch.keys():
					if key in new_outputs.batch:
						if isinstance(updated_outputs.batch[key], torch.Tensor):
							updated_outputs.batch[key][void_indices] = new_outputs.batch[key]
						else:
							# Handle list-like data
							for i, void_idx in enumerate(void_indices):
								updated_outputs.batch[key][void_idx] = new_outputs.batch[key][i]
		
		# Handle non-tensor batch data
		if original_outputs.non_tensor_batch is not None:
			updated_outputs.non_tensor_batch = {}
			for key, data in original_outputs.non_tensor_batch.items():
				if isinstance(data, list):
					updated_outputs.non_tensor_batch[key] = data.copy()
				elif hasattr(data, 'clone'):
					updated_outputs.non_tensor_batch[key] = data.clone()
				else:
					updated_outputs.non_tensor_batch[key] = data
			
			# Update void indices with new data
			if new_outputs.non_tensor_batch is not None:
				for key in updated_outputs.non_tensor_batch.keys():
					if key in new_outputs.non_tensor_batch:
						if isinstance(updated_outputs.non_tensor_batch[key], list):
							for i, void_idx in enumerate(void_indices):
								updated_outputs.non_tensor_batch[key][void_idx] = new_outputs.non_tensor_batch[key][i]
						elif hasattr(updated_outputs.non_tensor_batch[key], '__setitem__'):
							updated_outputs.non_tensor_batch[key][void_indices] = new_outputs.non_tensor_batch[key]
		
		return updated_outputs

	def rollout(self, dataproto: DataProto, val=False):
		es_manager = self.val_es_manager if val else self.train_es_manager
		ctx_manager = self.val_ctx_manager if val else self.train_ctx_manager
		env_outputs = es_manager.reset()

		for i in range(self.config.agent_proxy.max_turn):
			lm_inputs: DataProto = ctx_manager.get_lm_inputs(env_outputs, prepare_for_update=False)
			
			lm_inputs.meta_info = dataproto.meta_info # TODO: setup vllm early stop when max length is reached. make sure this can be done

			#* vanilla rollout of LLM
			lm_outputs: DataProto = self.generate_sequences(lm_inputs,val=val)
			
			#* Check for void actions and regenerate if necessary
			if val==False:
				lm_outputs = self._handle_void_actions(lm_inputs, lm_outputs)

			#* env_inputs: manage context in a list ['env_id', 'llm_raw_response', 'llm_response', 'actions'], len is mini_bs
			#* 'actions':  Only the first MAX_ACTIONS actions are kept in the rollout
			#* "llm_response" transforms "llm_raw_response" into <think>xx</think><answer>xx</answer>
			env_inputs: List[Dict] = ctx_manager.get_env_inputs(lm_outputs)
			env_outputs: List[Dict] = es_manager.step(env_inputs)
			if len(env_outputs) == 0: # all finished
				break
		rollout_states = es_manager.get_rollout_states() 
		rollouts = ctx_manager.formulate_rollouts(rollout_states)
		# self.tokenizer.batch_decode(rollouts.batch['input_ids'], skip_special_tokens=False) # see all the trajectories
		return rollouts

@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config):
	# detect config name from python -m ragen.llm_agent.agent_proxy --config_name frozen_lake
	os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
	os.environ["CUDA_VISIBLE_DEVICES"] = str(config.system.CUDA_VISIBLE_DEVICES)
	tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
	actor_wg = VllmWrapperWg(config, tokenizer)
	proxy = LLMAgentProxy(config, actor_wg, tokenizer)
	import time
	for _ in range(3):
		start_time = time.time()
		rollouts = proxy.rollout(DataProto(batch=None, non_tensor_batch=None, meta_info={'eos_token_id': 151645, 'pad_token_id': 151643, 'recompute_log_prob': False, 'do_sample':config.actor_rollout_ref.rollout.do_sample, 'validate': True}), val=True)
		end_time = time.time()
		print(f'rollout time: {end_time - start_time} seconds')
		# print rollout rewards from the rm_scores
		rm_scores = rollouts.batch["rm_scores"]
		metrics = rollouts.meta_info["metrics"]
		avg_reward = rm_scores.sum(-1).mean().item()
		print(f'rollout rewards: {avg_reward}')
		print(f'metrics:')
		for k, v in metrics.items():
			print(f'{k}: {v}')

# @hydra.main(version_base=None, config_path="../../config", config_name="evaluate_api_llm")
# def main(config):
# 	# detect config name from python -m ragen.llm_agent.agent_proxy --config_name frozen_lake
# 	tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
# 	actor_wg = ApiCallingWrapperWg(config, tokenizer)
# 	proxy = LLMAgentProxy(config, actor_wg, tokenizer)
# 	import time
# 	start_time = time.time()
# 	rollouts = proxy.rollout(DataProto(batch=None, non_tensor_batch=None, meta_info={'eos_token_id': 151645, 'pad_token_id': 151643, 'recompute_log_prob': False, 'do_sample': False, 'validate': True}), val=True)
# 	print(f'[DEBUG] rollouts: {rollouts}')
# 	end_time = time.time()
# 	print(f'rollout time: {end_time - start_time} seconds')
# 	# print rollout rewards from the rm_scores
# 	rm_scores = rollouts.batch["rm_scores"]
# 	metrics = rollouts.meta_info["metrics"]
# 	avg_reward = rm_scores.sum(-1).mean().item()
# 	print(f'rollout rewards: {avg_reward}')
# 	print(f'metrics:')
# 	for k, v in metrics.items():
# 		print(f'{k}: {v}')



if __name__ == "__main__":
	main()
