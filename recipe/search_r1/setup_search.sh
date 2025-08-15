set -e

eval "$(conda shell.bash hook)"
conda activate verl

python3 -m uv pip install bitsandbytes deepspeed==0.16.4 isort jsonlines loralib optimum tensorboard torchmetrics transformers_stream_generator
python3 -m uv pip install llama_index bs4 pymilvus infinity_client omegaconf hydra-core easydict mcp==1.9.3

python3 -m uv pip install "faiss-gpu-cu12==1.9.0"
# pip install -U "numpy==2.2.6"
python3 -m uv pip install nvidia-cublas-cu12==12.4.5.8 

save_path=rag_server_dir
python recipe/search_r1/rag_server/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz

python recipe/search_r1/rag_server/data_process/nq_search.py

