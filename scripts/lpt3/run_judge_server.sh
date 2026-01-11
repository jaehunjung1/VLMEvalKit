MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507

NUM_GPUS=$(nvidia-smi -L | wc -l)
vllm serve $MODEL --port 8000 --host 0.0.0.0 --dtype bfloat16 --tensor-parallel-size "$NUM_GPUS" --api-key "vlmevalkit" --max-model-len 20k