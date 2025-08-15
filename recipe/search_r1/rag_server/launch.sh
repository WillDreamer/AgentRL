

nvcc --version
# export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
# export LD_LIBRARY_PATH=/usr/local/cuda/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

file_path=rag_server_dir
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever=intfloat/e5-base-v2

python3 recipe/search_r1/rag_server/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_model $retriever &
sleep 12000
