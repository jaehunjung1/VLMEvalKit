# Qwen/Qwen2.5-VL-7B-Instruct csfufu/Revisual-R1-final XiaomiMiMo/MiMo-VL-7B-SFT XiaomiMiMo/MiMo-VL-7B-RL Qwen/Qwen3-VL-8B-Instruct

#MODEL=Qwen/Qwen3-VL-8B-Instruct
MODEL=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/lpt/lpt3-sft/scripts/lpt/checkpoints/qwen3_7b--v2_charts--lr5e-6/checkpoint-500

vllm serve $MODEL --port 8000 --host 0.0.0.0 --dtype bfloat16 --tensor-parallel-size 4 \
--allowed-local-media-path / --limit-mm-per-prompt '{"image": 16}'