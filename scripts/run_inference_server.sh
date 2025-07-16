#vllm serve Qwen/Qwen2.5-VL-3B-Instruct --port 8000 --host 0.0.0.0 --dtype bfloat16 --tensor-parallel-size 4 \
#--allowed-local-media-path / --limit-mm-per-prompt image=10

#vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000 --host 0.0.0.0 --dtype bfloat16 --tensor-parallel-size 4 \
#--allowed-local-media-path / --limit-mm-per-prompt image=10

vllm serve csfufu/Revisual-R1-final --port 8000 --host 0.0.0.0 --dtype bfloat16 --tensor-parallel-size 4 \
--allowed-local-media-path / --limit-mm-per-prompt image=10