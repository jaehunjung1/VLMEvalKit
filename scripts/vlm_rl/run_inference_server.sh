# Qwen/Qwen2.5-VL-7B-Instruct csfufu/Revisual-R1-final XiaomiMiMo/MiMo-VL-7B-SFT XiaomiMiMo/MiMo-VL-7B-RL

MODEL=XiaomiMiMo/MiMo-VL-7B-RL
#MODEL=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/verl/verl-projects/vlm_rl/checkpoints/mixture_v2-l32k/hf_global_step_200

vllm serve $MODEL --port 8000 --host 0.0.0.0 --dtype bfloat16 --tensor-parallel-size 4 \
--allowed-local-media-path / --limit-mm-per-prompt '{"image": 16}'