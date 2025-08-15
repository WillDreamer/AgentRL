set -e

eval "$(conda shell.bash hook)"
conda activate verl

pip3 install accelerate bitsandbytes datasets deepspeed==0.16.4 einops flash-attn==2.7.0.post2 isort jsonlines loralib optimum packaging peft pynvml>=12.0.0 ray[default]==2.46.0 tensorboard torch==2.6.0 torchmetrics tqdm transformers==4.51.3 transformers_stream_generator wandb wheel
pip install "qwen-agent[code_interpreter]"
pip install llama_index bs4 pymilvus infinity_client codetiming tensordict==0.6 omegaconf torchdata==0.10.0 hydra-core easydict dill python-multipart mcp==1.9.3
sudo apt-get install -y nvidia-cuda-toolkit
pip install "faiss-gpu-cu12==1.9.0"
# pip install -U "numpy==2.2.6"
pip install nvidia-cublas-cu12==12.4.5.8 

save_path=rag_server
python recipe/search_r1/rag_server/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz

python recipe/search_r1/rag_server/data_process/nq_search.py

