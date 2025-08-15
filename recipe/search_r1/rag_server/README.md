# demo of search-R1

## env configuration
```bash
conda create -n qwen_demo python=3.10
conda activate searchr1
pip3 install torch-2.6.0 torchaudio-2.6.0 torchvision-0.21.0
pip3 install vllm==0.8.5 

# flash attention 2
pip3 install flash-attn --no-build-isolation 
pip3 install faiss-gpu-cu12 uvicorn fastapi mcp #这里也可以安装faiss-gpu==1.8
```
## Quick start

(1) download dataset
```bash
save_path=/your/path/to/save
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

(2) process NQ dataset.
```bash
python scripts/nq_search.py
```
