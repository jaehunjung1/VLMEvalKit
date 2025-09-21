MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
vllm serve $MODEL --port 8000 --host 0.0.0.0 --dtype bfloat16 --tensor-parallel-size 8 --api-key "vlmevalkit" --max-model-len 20k