#MODEL_NAME=Jaehun/lpt2-stage2-sft-dpo
MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct

vllm serve $MODEL_NAME --port 8000 --host 0.0.0.0 --dtype bfloat16 --tensor-parallel-size 4 \
--allowed-local-media-path / --limit-mm-per-prompt '{"image": 16}'