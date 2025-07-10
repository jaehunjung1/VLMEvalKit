export AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4o-20241120"
export OPENAI_API_VERSION="2025-02-01-preview"

## for 3B, we need to use 4 devices
cd ..
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --data MathVista_MINI --model Qwen2.5-VL-3B-Instruct --use-vllm --api-nproc 8 --retry 20 --judge gpt-4o-mini --reuse --judge-args "{\"use_azure\": true}" --verbose


