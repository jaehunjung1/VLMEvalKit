export VLLM_API_BASE="http://0.0.0.0:8000/v1/chat/completions"
export AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4o-20241120"
export OPENAI_API_VERSION="2025-02-01-preview"

# Qwen2.5-VL-3B-VLLM, Qwen2.5-VL-7B-VLLM, ReVisual-R1-VLLM, MiMo-VL-7B-SFT-VLLM, MiMo-VL-7B-RL-VLLM, VLAA-Thinker-Qwen2.5VL-7B-VLLM
# custom models: lpt2-stage2-sft-dpo
#MODEL=lpt2-dpo_70k-sft_247k-hf_global_step_460

MODEL=/lustre/fsw/portfolios/nvr/users/dacunamarrer/lptv2/output/stage2/dpo_fixeval_new_only_stage2_fixeval_harden_distill72b_671b_v2_objpt_750k_acc_7904_checkpoint-8100/checkpoint-650
SAVE_DIR_NAME=dpo_fixeval_new_only_stage2_fixeval_harden_distill72b_671b_v2_objpt_750k_acc_7904_checkpoint-8100_checkpoint-650

# VStarBench MMVP RealWorldQA CV-Bench-2D MMStar_filtered MME-RealWorld-Filtered
DATA="VStarBench MMVP RealWorldQA CV-Bench-2D MMStar_filtered"
JUDGE=gpt-4o-mini


# wait until inference server is ready
while ! nc -z 0.0.0.0 8000; do
  echo "[INFO] Waiting for inference server to accept connections..."
  sleep 10s
done
echo "[INFO] Inference server is ready to accept connections!"


cd ../..
if [ -n "$SAVE_DIR_NAME" ]; then
  python run.py --data $DATA --save_dir_name $SAVE_DIR_NAME \
  --model $MODEL --infer-api-nproc 512 --infer-retry 3 \
  --judge $JUDGE --judge-api-nproc 2 --judge-retry 10 --judge-args "{\"use_azure\": true}" --work-dir "./outputs/lpt2" \
  --reuse --verbose
else
  python run.py --data $DATA \
  --model $MODEL --infer-api-nproc 512 --infer-retry 3 \
  --judge $JUDGE --judge-api-nproc 2 --judge-retry 10 --judge-args "{\"use_azure\": true}" --work-dir "./outputs/lpt2" \
  --reuse --verbose
fi

