vllm serve Qwen/Qwen2.5-VL-3B-Instruct --port 8000 --host 0.0.0.0 --dtype bfloat16 --tensor-parallel-size 4 \
--allowed-local-media-path / --limit-mm-per-prompt image=10