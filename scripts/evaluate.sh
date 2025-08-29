export VLLM_API_BASE="http://0.0.0.0:8000/v1/chat/completions"
export AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4o-20241120"
export OPENAI_API_VERSION="2025-02-01-preview"

# Qwen2.5-VL-3B-VLLM, Qwen2.5-VL-7B-VLLM, ReVisual-R1-VLLM, MiMo-VL-7B-SFT-VLLM, MiMo-VL-7B-RL-VLLM
MODEL=MiMo-VL-7B-SFT-VLLM
# MathVista_MINI, MMMU_Pro_10c LogicVista, RealWorldQA, HallusionBench
DATA=MMMU_Pro_10c
JUDGE=gpt-4o-mini

cd ..
python run.py --data $DATA \
--model $MODEL --infer-api-nproc 32 --infer-retry 3 \
--judge $JUDGE --judge-api-nproc 8 --judge-retry 10 --judge-args "{\"use_azure\": true}" \
--reuse --verbose

