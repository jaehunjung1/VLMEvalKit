export VLLM_API_BASE="http://0.0.0.0:8000/v1/chat/completions"
export AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4o-20241120"
export OPENAI_API_VERSION="2025-02-01-preview"

#export VLLM_TP_SIZE=4
# TODO start vllm server with 4 devices

## for 3B, we need to use 4 devices
cd ..
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --data MathVista_MINI \
--model Qwen2.5-VL-3B-VLLM --infer-api-nproc 32 --infer-retry 3 \
--judge gpt-4o-mini --judge-api-nproc 8 --judge-retry 10 --judge-args "{\"use_azure\": true}" \
--reuse --verbose


