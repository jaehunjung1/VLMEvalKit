#MODEL_NAME=Jaehun/lpt2-stage2-sft-dpo
#MODEL_NAME=Jaehun/lpt2-acc-6414-sft-750k-checkpoint-2750
#MODEL_NAME=Jaehun/lpt2-acc-6632-dpo-130k-sft-247k-checkpoint-250
#MODEL_NAME=Jaehun/lpt2-dpo_70k-sft_247k-hf_global_step_460
#MODEL_NAME=Jaehun/lpt2-stage2_distill72b_671b_v2__sft_docci_objpt_247k_train_acc7511_checkpoint-2900
#MODEL_NAME=Jaehun/lpt2-dpo_distill72b_671b_v2__sft_docci_objpt_247k_train_acc7445_acc7589_checkpoint-500
#MODEL_NAME=/lustre/fsw/portfolios/nvr/users/dacunamarrer/lptv2/ckpts_stage2_sft/dpo_fixeval_new_only_stage2_fixeval_harden_distill72b_671b_v2_objpt_750k_acc_7904_checkpoint-8100_acc_939_checkpoint-350
#MODEL_NAME=/lustre/fsw/portfolios/nvr/users/dacunamarrer/lptv2/output/stage2/dpo_stage_2_fixeval_sft_stage2_fixeval_combined_harden_distill72b_671b_v2_objpt_750k_acc_6766_checkpoint-2500/checkpoint-750

MODEL_NAME=andrewliao11/LongPerceptualThought-SFT_then_DPO
#MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct
#MODEL_NAME=UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-7B
#MODEL_NAME=csfufu/Revisual-R1-final
#MODEL_NAME=XiaomiMiMo/MiMo-VL-7B-SFT


vllm serve $MODEL_NAME --port 8000 --host 0.0.0.0 --dtype bfloat16 --tensor-parallel-size 4 --max-model-len 20k \
--allowed-local-media-path / --limit-mm-per-prompt '{"image": 16}'