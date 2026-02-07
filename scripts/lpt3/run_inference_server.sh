# Qwen/Qwen2.5-VL-7B-Instruct csfufu/Revisual-R1-final XiaomiMiMo/MiMo-VL-7B-SFT XiaomiMiMo/MiMo-VL-7B-RL Qwen/Qwen3-VL-8B-Instruct Qwen/Qwen3-VL-8B-Thinking, nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16

MODEL=Qwen/Qwen3-VL-8B-Thinking
#MODEL=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/lpt/lpt3-sft/scripts/lpt/checkpoints/hrv--v2_hr-pdmw--curate-rwqa_v1--lr5e-6/checkpoint-200
#MODEL=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/lpt/verl-opd/projects/lpt3/checkpoints/debug-235B-8B-thinking-N4-k2-16k/hf_global_step_140
#MODEL=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/lpt/verl-opd/projects/lpt3/checkpoints/ablation--sim_skill_10000-10000/hf_global_step_75

vllm serve $MODEL --port 8000 --host 0.0.0.0 --dtype bfloat16 --tensor-parallel-size 4 \
--allowed-local-media-path / --limit-mm-per-prompt '{"image": 4, "video": 0}' --max-model-len 65536 --trust_remote_code
