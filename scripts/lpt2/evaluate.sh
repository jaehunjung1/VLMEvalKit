export VLLM_API_BASE="http://0.0.0.0:8000/v1/chat/completions"
export AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4o-20241120"
export OPENAI_API_VERSION="2025-02-01-preview"

# Qwen2.5-VL-3B-VLLM, Qwen2.5-VL-7B-VLLM, ReVisual-R1-VLLM, MiMo-VL-7B-SFT-VLLM, MiMo-VL-7B-RL-VLLM, lpt2-stage2-sft-dpo
MODEL=lpt2-acc-6414-sft-750k-checkpoint-2750

# MathVista_MINI, MMMU_Pro_10c LogicVista, RealWorldQA, HallusionBench
DATA="RealWorldQA"
JUDGE=gpt-4o-mini


# wait until inference server is ready
while ! nc -z 0.0.0.0 8000; do
  echo "[INFO] Waiting for judge server to accept connections..."
  sleep 10s
done
echo "[INFO] Judge server is ready to accept connections!"


cd ../..
python run.py --data $DATA \
--model $MODEL --infer-api-nproc 512 --infer-retry 3 \
--judge $JUDGE --judge-api-nproc 4 --judge-retry 10 --judge-args "{\"use_azure\": true}" \
--reuse --verbose

