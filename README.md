<div align="center">
 👋 Hi, everyone! 
    We will tell the world open-source agentic RL area is suffering!
    <br>
    <br>
</div>


<h1 style="text-align: center;"> 🤖 Agentic RL Arena </h1>


<p align="center"><img src="public/ARL.png" width="800px" alt="Agentic RL Arena Framework" /></p>
<p align="center" style="font-size: 16px; max-width: 800px; margin: 0 auto;">
Figure 1: Overview of our framework
</p>

## Our Framework Design

- Our benchmark profoundly analyzes the existing sufferings of Agentic RL, including tool-intergrated and multi-turn RL.
- Our benchmark comprehensively compares existing Agentic RL algorithms and their breakthroughs.
- Our benchmark universally provides experimental results and findings on multiple ARC tasks.

### 🔥 Key Features

- ✅ Support Multi-turn Math+Code Interpreter Agents
- ✅ Support Multi-turn Game Agents
- ✅ Support Multi-turn Shopping Agents

### 🔧 Upcoming Features and Changes

- ➡️ Support WebAgents
- ➡️ Support Multimodal Agents
- ➡️ Support Software Enginnering Agents

### 📅 TODO

- [ ] Cross-domain agentic reasoning
- [ ] Multiple tool integration reasoning
- [ ] Gap between $\pi_{\theta_{vLLM}}$ and $\pi_{\theta_{FSDP}}$
- [ ] Turn-level rollout compactness
- [ ] Dynamical sampling temperature
- [x] Rollout analysis

## 💡 Getting Started


Our benchmark is based on the following main dependencies:

```python
Python=3.11, VeRL=0.4.0, PyTorch=2.6.0, and vLLM=0.8.5
```

You can install other requirements as follows:
``` bash
# Install conda (Optional if conda exists)
bash set_conda.sh

# Install foundational dependencies 
bash setup_env.sh

# Install dependencies for specific tasks
conda activate verl
pip install -r requirements_xxx.txts
```

## 🌊 Easy Extension

🔹 All of the methods utilized is in `recipe`, you can warp the verl worker for your code to join our codebase. The folder under `recipe` can represent either a method for different tasks or a series methods for one task. You can refer to [Easy Extension](docs/extension.md) for examples.

🔹 Add specific dependencies to `requirements_xxx.txt`

🔹 Feel free to add the folder of the third-party tools, e.g., `AgentRL/sandbox` for code implementation.


## 🚀 Existing Support

> ### 🧮 **Math+CI**


1. We use Sandbox Fusion as an asynchronous code interpreter. You can follow the [Guidance](sandbox/README.md) to run the CI.

2. The training datasets are Math3-5 from SimpleRL and Deepscaler in `datasets`.

```bash
# 3. Run the demo code with:
conda activate verl
bash train_simpletir.sh
```

> ### 🎮 **OpenAI Game**

```python
# 1. Install the additional dependecy
conda activate verl
pip install -r requirements_ragen.txt

# 2. Download the data
python recipe/ragen/download_data.py

# 3. Run the demo code with:
bash train_ragen.sh
```

> ### 🛒 **Shopping Agents**

```bash
# 1. Build the webshop environments
bash recipe/webshop/setup_webshop.sh

# 2. Run the demo code with:
conda activate verl
bash train_webshop.sh
```

> ### 🕸️ **Search Agents**

```bash
#! 1. Build the RAG server environments
bash recipe/search_r1/setup_search.sh

#! 2. Run the RAG Server: Before running the bash script, modify the parameters in launch.sh (`file_path` is the storage location of the corpus, `retriever` is the local path of the model intfloat/e5-base-v2)
bash recipe/search_r1/rag_server/launch.sh

# 3. Run the demo code with:
bash train_search.sh
```

## 📊 Further Analysis

### Visualization of rollout

1. Utilize the `custom_reward_function` to define a specific file, you can refer to `DummyRewardManager` in `recipe/webshop/main_webshop.py` for example.

2. Add the hyperpapram in the training scripts as follows:

```bash

# claim the saving folder
export ROLLOUT_DUMP_DIR=/workspace/AgentRL/outputs/
export ROLLOUT_DUMP_ENABLE=True
export ROLLOUT_SAVE_EVERY_STEPS=10

# Merge all of the generated seq
cd /abs/path/rollouts
cat rollouts_*.jsonl > all_rollouts.jsonl
```


<!-- 

```bash
# Install requiremnet
pip install mlflow

# Start server
mlflow server \
  --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:////tmp/mlruns.db \
  --default-artifact-root /tmp/mlruns

export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

actor_rollout_ref.rollout.trace.backend: mlflow  # or weave
actor_rollout_ref.rollout.trace.token2text: True
trainer.logger: ['console', 'mlflow']
``` -->

## 🎆 Awesome work for reference

- [TinyZero](https://github.com/Jiayi-Pan/TinyZero): a reproduction of **DeepSeek R1 Zero** recipe for reasoning tasks ![GitHub Repo stars](https://img.shields.io/github/stars/Jiayi-Pan/TinyZero)
- [SkyThought](https://github.com/NovaSky-AI/SkyThought): RL training for Sky-T1-7B by NovaSky AI team. ![GitHub Repo stars](https://img.shields.io/github/stars/NovaSky-AI/SkyThought)
- [simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason): SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild ![GitHub Repo stars](https://img.shields.io/github/stars/hkust-nlp/simpleRL-reason)
- [Easy-R1](https://github.com/hiyouga/EasyR1): **Multi-modal** RL training framework ![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/EasyR1)
- [OpenManus-RL](https://github.com/OpenManus/OpenManus-RL): LLM Agents RL tunning framework for multiple agent environments. ![GitHub Repo stars](https://img.shields.io/github/stars/OpenManus/OpenManus-RL)
- [rllm](https://github.com/agentica-project/rllm): async RL training with [verl-pipeline](https://github.com/agentica-project/verl-pipeline) ![GitHub Repo stars](https://img.shields.io/github/stars/agentica-project/rllm)
- [PRIME](https://github.com/PRIME-RL/PRIME): Process reinforcement through implicit rewards ![GitHub Repo stars](https://img.shields.io/github/stars/PRIME-RL/PRIME)
- [RAGEN](https://github.com/ZihanWang314/ragen): a general-purpose reasoning **agent** training framework ![GitHub Repo stars](https://img.shields.io/github/stars/ZihanWang314/ragen)
- [Logic-RL](https://github.com/Unakar/Logic-RL): a reproduction of DeepSeek R1 Zero on 2K Tiny Logic Puzzle Dataset. ![GitHub Repo stars](https://img.shields.io/github/stars/Unakar/Logic-RL)
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1): RL with reasoning and **searching (tool-call)** interleaved LLMs ![GitHub Repo stars](https://img.shields.io/github/stars/PeterGriffinJin/Search-R1)
- [DeepRetrieval](https://github.com/pat-jj/DeepRetrieval): RL Training of **Search Agent** with **Search/Retrieval Outcome** ![GitHub Repo stars](https://img.shields.io/github/stars/pat-jj/DeepRetrieval)
- [ReSearch](https://github.com/Agent-RL/ReSearch): Learning to **Re**ason with **Search** for LLMs via Reinforcement Learning ![GitHub Repo stars](https://img.shields.io/github/stars/Agent-RL/ReSearch)
- [Code-R1](https://github.com/ganler/code-r1): Reproducing R1 for **Code** with Reliable Rewards ![GitHub Repo stars](https://img.shields.io/github/stars/ganler/code-r1)
- [Skywork-OR1](https://github.com/SkyworkAI/Skywork-OR1): Skywork open reaonser series ![GitHub Repo stars](https://img.shields.io/github/stars/SkyworkAI/Skywork-OR1)
- [ToRL](https://github.com/GAIR-NLP/ToRL): Scaling tool-integrated RL ![GitHub Repo stars](https://img.shields.io/github/stars/GAIR-NLP/ToRL)
- [verl-agent](https://github.com/langfengQ/verl-agent): A scalable training framework for **long-horizon LLM/VLM agents**, along with a new algorithm **GiGPO** ![GitHub Repo stars](https://img.shields.io/github/stars/langfengQ/verl-agent)
- [PF-PPO](https://arxiv.org/abs/2409.06957): Policy Filtration for PPO based on the reliability of reward signals for more efficient and robust RLHF.
- [GUI-R1](https://github.com/ritzz-ai/GUI-R1): **GUI-R1**: A Generalist R1-style Vision-Language Action Model For **GUI Agents** ![GitHub Repo stars](https://img.shields.io/github/stars/ritzz-ai/GUI-R1)
- [DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher): Scaling deep research via reinforcement learning in real-world environments ![GitHub Repo stars](https://img.shields.io/github/stars/GAIR-NLP/DeepResearcher)
- [VAGEN](https://github.com/RAGEN-AI/VAGEN): Training VLM agents with multi-turn reinforcement learning ![GitHub Repo stars](https://img.shields.io/github/stars/RAGEN-AI/VAGEN)
- [ReTool](https://retool-rl.github.io/): ReTool: reinforcement learning for strategic tool use in LLMs
- [Seed-Coder](https://github.com/ByteDance-Seed/Seed-Coder): RL training of Seed-Coder boosts performance on competitive programming ![GitHub Repo stars](https://img.shields.io/github/stars/ByteDance-Seed/Seed-Coder)
- [all-hands/openhands-lm-32b-v0.1](https://www.all-hands.dev/blog/introducing-openhands-lm-32b----a-strong-open-coding-agent-model): A strong, open coding agent model, trained with [multi-turn fine-tuning](https://github.com/volcengine/verl/pull/195)
- [RM-R1](https://arxiv.org/abs/2505.02387): RL training of reasoning reward models ![GitHub Repo stars](https://img.shields.io/github/stars/RM-R1-UIUC/RM-R1)
- [Absolute Zero Reasoner](https://arxiv.org/abs/2505.03335): A no human curated data self-play framework for reasoning![GitHub Repo stars](https://img.shields.io/github/stars/LeapLabTHU/Absolute-Zero-Reasoner)
- [LUFFY](https://arxiv.org/pdf/2504.14945): Learning to Reason under Off-Policy Guidance![GitHub Repo stars](https://img.shields.io/github/stars/ElliottYan/LUFFY)
- [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool): An unified and easy-to-extend tool-agent training framework based on verl![GitHub Repo stars](https://img.shields.io/github/stars/TIGER-AI-Lab/verl-tool)
- [DeepMath](https://github.com/zwhe99/DeepMath): DeepMath-103K data and series models for math reasoning![GitHub Repo stars](https://img.shields.io/github/stars/zwhe99/DeepMath)
