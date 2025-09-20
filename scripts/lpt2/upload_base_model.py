from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor


if __name__ == "__main__":
    # model_path = Path("/lustre/fsw/portfolios/nvr/users/dacunamarrer/lptv2/lpt-QwenVL-25-ckpt/acc-6414-sft-750k-checkpoint-2750")
    # hf_model_name = f"Jaehun/lpt2-{model_path.stem}"
    # model_path = Path("/lustre/fsw/portfolios/nvr/users/dacunamarrer/lptv2/lpt-QwenVL-25-ckpt/acc-6632-dpo-130k-sft-247k-checkpoint-250")
    # hf_model_name = f"Jaehun/lpt2-{model_path.stem}"
    # model_path = Path("/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/verl/verl/adaptations/lpt2/checkpoints/dpo_70k-sft_247k/hf_global_step_460")
    # hf_model_name = f"Jaehun/lpt2-dpo_70k-sft_247k-hf_global_step_460"
    # model_path = Path("/lustre/fsw/portfolios/nvr/users/dacunamarrer/lptv2/ckpts_stage2_sft/stage2_distill72b_671b_v2__sft_docci_objpt_247k_train_acc7511_checkpoint-2900")
    # hf_model_name = f"Jaehun/lpt2-{model_path.stem}"
    model_path = Path("/lustre/fsw/portfolios/nvr/users/dacunamarrer/lptv2/ckpts_stage2_sft/dpo_distill72b_671b_v2__sft_docci_objpt_247k_train_acc7445_acc7589_checkpoint-500")
    hf_model_name = f"Jaehun/lpt2-{model_path.stem}"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="bfloat16")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)

    model.push_to_hub(hf_model_name)
    tokenizer.push_to_hub(hf_model_name)
    processor.push_to_hub(hf_model_name)




