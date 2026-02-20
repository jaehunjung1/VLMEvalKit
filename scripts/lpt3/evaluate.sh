## === using VLLM local API === #
#export VLLM_API_BASE="http://0.0.0.0:8000/v1/chat/completions"
#export JUDGE_API_NODE="nvl72045-T18"  # todo
#JUDGE=Qwen3-30B-A3B-Instruct-2507
#
## wait until inference & judge server is ready
#while ! nc -z 0.0.0.0 8000; do
#  echo "[INFO] Waiting for inference server to accept connections..."
#  sleep 10s
#done
#echo "[INFO] Inference server is ready to accept connections!"
#
#
#while ! nc -z $JUDGE_API_NODE 8000; do
#  echo "[INFO] Waiting for judge server to accept connections..."
#  sleep 10s
#done
#echo "[INFO] Judge server is ready to accept connections!"
#
## Qwen2.5-VL-3B-VLLM, Qwen2.5-VL-7B-VLLM, ReVisual-R1-VLLM, MiMo-VL-7B-SFT-VLLM, MiMo-VL-7B-RL-VLLM, Qwen3-VL-8B-Instruct-VLLM, Qwen3-VL-8B-Thinking-VLLM, NVIDIA-Nemotron-Nano-12B-v2-VL-BF16
##MODEL=Qwen3-VL-8B-Thinking-VLLM
##SAVE_DIR_NAME=Qwen3-VL-8B-Thinking-VLLM
#
##MODEL=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/lpt/lpt3-sft/scripts/lpt/checkpoints/hrv--v2_hr-pdmw--curate-rwqa_v1--lr5e-6/checkpoint-200
##SAVE_DIR_NAME=hrv--v2_hr-pdmw--curate-rwqa_v1--lr5e-6--checkpoint-200
#
#MODEL=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/lpt/verl-opd/projects/lpt3/checkpoints/debug-235B-8B-thinking-N4-k2-16k/hf_global_step_140
#SAVE_DIR_NAME=debug-235B-8B-thinking-N4-k2-16k--hf_global_step_140
#
## CharXiv_reasoning_val RealWorldQA HRBench4K VStarBench ZEROBench ZEROBench_sub InfoVQA_VAL SEEDBench2_Plus CV-Bench-2D StaticEmbodiedBench MMVP HallusionBench CRPE_RELATION
#DATA="CharXiv_reasoning_val"
#
## run evaluation
#cd ../..
#python run.py --data $DATA --save_dir_name $SAVE_DIR_NAME \
#--model $MODEL --infer-api-nproc 512 --infer-retry 3 \
#--judge $JUDGE --judge-api-nproc 128 --judge-retry 10 --work-dir "./outputs/lpt3" \
#--reuse --verbose


# === using inference.nvidia.com === #
export VLLM_API_BASE="http://0.0.0.0:8000/v1/chat/completions"
JUDGE=gpt-4o-mini

# wait until inference & judge server is ready
while ! nc -z 0.0.0.0 8000; do
  echo "[INFO] Waiting for inference server to accept connections..."
  sleep 10s
done
echo "[INFO] Inference server is ready to accept connections!"

# Qwen2.5-VL-3B-VLLM, Qwen2.5-VL-7B-VLLM, ReVisual-R1-VLLM, MiMo-VL-7B-SFT-VLLM, MiMo-VL-7B-RL-VLLM, Qwen3-VL-8B-Instruct-VLLM, Qwen3-VL-8B-Thinking-VLLM, NVIDIA-Nemotron-Nano-12B-v2-VL-BF16
#MODEL=Qwen3-VL-8B-Thinking-VLLM
#SAVE_DIR_NAME=Qwen3-VL-8B-Thinking-VLLM

#MODEL=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/lpt/lpt3-sft/scripts/lpt/checkpoints/hrv--v2_hr-pdmw--curate-rwqa_v1--lr5e-6/checkpoint-200
#SAVE_DIR_NAME=hrv--v2_hr-pdmw--curate-rwqa_v1--lr5e-6--checkpoint-200

MODEL=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/lpt/verl-opd/projects/lpt3/checkpoints/unified_v1_180k_235B_v2/hf_global_step_240
SAVE_DIR_NAME=unified_v1_180k_235B_v2--hf_global_step_240

# CharXiv_reasoning_val RealWorldQA VStarBench SEEDBench2_Plus HRBench4K ZEROBench ZEROBench_sub InfoVQA_VAL CV-Bench-2D StaticEmbodiedBench MMVP HallusionBench CRPE_RELATION MathVista_MINI LogicVista MMMU_Pro_10c
DATA="MMMU_Pro_10c"

# run evaluation
cd ../..
python run.py --data $DATA --save_dir_name $SAVE_DIR_NAME \
--model $MODEL --infer-api-nproc 512 --infer-retry 3 \
--judge $JUDGE --judge-api-nproc 8 --judge-retry 10 --work-dir "./outputs/lpt3" \
--reuse --verbose





## === using Azure API === #
#export VLLM_API_BASE="http://0.0.0.0:8000/v1/chat/completions"
#export AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini-20240718  # "gpt-4o-20241120"
#export OPENAI_API_VERSION="2025-02-01-preview"
#JUDGE=gpt-4o-mini
#
## Qwen2.5-VL-3B-VLLM, Qwen2.5-VL-7B-VLLM, ReVisual-R1-VLLM, MiMo-VL-7B-SFT-VLLM, MiMo-VL-7B-RL-VLLM
#MODEL=MiMo-VL-7B-SFT-VLLM
#SAVE_DIR_NAME=MiMo-VL-7B-SFT-VLLM
#
##MODEL=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/verl/verl/adaptations/vlm_rl/checkpoints/mixture1_1-kl0.001-ent0/hf_global_step_340
##SAVE_DIR_NAME=mixture1_1-kl0.001-ent0--hf_global_step_340
#
#MODEL=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/verl/verl-projects/vlm_rl/checkpoints/physreason-ugphysics/hf_global_step_100
#SAVE_DIR_NAME=physreason-ugphysics--hf_global_step_100
#
## MathVerse_MINI_Vision_Only
#DATA=MathVerse_MINI_Vision_Only
#
## wait until inference server is ready
#while ! nc -z 0.0.0.0 8000; do
#  echo "[INFO] Waiting for inference server to accept connections..."
#  sleep 10s
#done
#echo "[INFO] Inference server is ready to accept connections!"
#
## run evaluation
#cd ../..
#python run.py --data $DATA --save_dir_name $SAVE_DIR_NAME \
#--model $MODEL --infer-api-nproc 512 --infer-retry 3 \
#--judge $JUDGE --judge-api-nproc 1 --judge-retry 10 --work-dir "./outputs/vlm_rl" --judge-args "{\"use_azure\": true}" \
#--reuse --verbose






#cd ../..

## run inference server
#PIDS=$(pgrep -f "vllm serve")
#if [ -n "$PIDS" ]; then
#    kill "$PIDS"
#else
#    echo "No vllm serving now."
#fi
#bash ./scripts/vlm_rl/run_inference_server.sh &> ./scripts/vlm_rl/logs/inference-server.log &
#
## wait until inference server is ready
#while ! nc -z 0.0.0.0 8000; do
#  echo "[INFO] Waiting for inference server to accept connections..."
#  sleep 10s
#done
#echo "[INFO] Inference server is ready to accept connections!"

## run inference
#python run.py --data $DATA --save_dir_name $SAVE_DIR_NAME \
#--model $MODEL --infer-api-nproc 32 --infer-retry 3 \
#--judge $JUDGE --judge-api-nproc 1 --judge-retry 10 --mode "infer" --work-dir "./outputs/vlm_rl" \
#--reuse --verbose

## run judge server
#kill "$(pgrep -f 'vllm serve')"
#sleep 5s
#bash ./scripts/vlm_rl/run_judge_server.sh &> ./scripts/vlm_rl/logs/judge-server.log &

## wait until inference server is ready
#while ! nc -z 0.0.0.0 8000; do
#  echo "[INFO] Waiting for judge server to accept connections..."
#  sleep 10s
#done
#echo "[INFO] Judge server is ready to accept connections!"

## run evaluation
#python run.py --data $DATA --save_dir_name $SAVE_DIR_NAME \
#--model $MODEL --infer-api-nproc 32 --infer-retry 3 \
#--judge $JUDGE --judge-api-nproc 1 --judge-retry 10 --work-dir "./outputs/vlm_rl" \
#--reuse --verbose



# todo if using azure api, we need to set --judge-args "{\"use_azure\": true}"
