import json
import re
import string
from pathlib import Path

import ipdb
import requests
from openai import OpenAI

from .base import BaseAPI
from .gpt import OpenAIWrapper
from ..vlm.qwen2_vl.model import ensure_video_url, ensure_image_url
from ..vlm.qwen2_vl.prompt import Qwen2VLPromptMixin


class Qwen2VLReasoningVLLM(BaseAPI, Qwen2VLPromptMixin):
    """Qwen-2-VL based Reasoning Models"""
    is_api: bool = True

    def __init__(self,
                 model_name: str,
                 api_base: str = None,
                 min_pixels: int = 1280 * 28 * 28,
                 max_pixels: int = 16384 * 28 * 28,
                 max_tokens: int = 32768,
                 temperature: float = 0.01,
                 retry: int = 10,
                 wait: int = 5,
                 timeout: int = 600,
                 limit_mm_per_prompt: int = 10,
                 verbose: bool = False,
                 **kwargs):

        self.model_name = model_name
        self.project_name = None
        if model_name == "ReVisual-R1-VLLM":
            self.full_model_name = "csfufu/Revisual-R1-final"
        elif model_name == "MiMo-VL-7B-SFT-VLLM":
            self.full_model_name = "XiaomiMiMo/MiMo-VL-7B-SFT"
        elif model_name == "MiMo-VL-7B-RL-VLLM":
            self.full_model_name = "XiaomiMiMo/MiMo-VL-7B-RL"
        elif model_name == "NVIDIA-Nemotron-Nano-12B-v2-VL-BF16":
            self.full_model_name = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
        elif model_name.startswith("lpt2-"):
            self.full_model_name = f"Jaehun/{self.model_name}"
        elif model_name == "LongPerceptualThought-SFT_then_DPO":
            self.full_model_name = "andrewliao11/LongPerceptualThought-SFT_then_DPO"
        elif Path(model_name).exists():
            # local checkpoint
            if "vlm_rl" in model_name:
                self.project_name = "vlm_rl"
            elif "lptv2" in model_name or "lpt2" in model_name:
                self.project_name = "lpt2"
            elif "lpt3" in model_name:
                self.project_name = "lpt3"
            else:
                ipdb.set_trace()
                raise NotImplementedError
            self.full_model_name = model_name
        elif model_name == "VLAA-Thinker-Qwen2.5VL-7B-VLLM":
            self.full_model_name = "UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-7B"
        else:
            ipdb.set_trace()
            raise NotImplementedError

        self.min_pixels, self.max_pixels = min_pixels, max_pixels

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retry, self.wait, self.timeout = retry, wait, timeout
        self.limit_mm_per_prompt = limit_mm_per_prompt

        if "system_prompt" in kwargs:
            self.config_system_prompt = kwargs.pop("system_prompt")
        else:
            self.config_system_prompt = None

        self.key = ""  # we won't set key for VLLM server

        BaseAPI.__init__(self, wait=wait, retry=retry, system_prompt=None, verbose=verbose, **kwargs)
        Qwen2VLPromptMixin.__init__(self, use_custom_prompt=False, **kwargs)

        self.api_base = api_base
        self.client = OpenAI(
            base_url=self.api_base,
            api_key=""
        )

        self.logger.info(f'Model: {self.full_model_name}; Using API Base: {self.api_base}')

    def use_custom_prompt(self, dataset: str) -> bool:
        # overriding Qwen2VLPromptMixing depending on the model
        if self.project_name == "lpt3":
            if dataset == "RealWorldQA":
                return True
            elif "CharXiv" in dataset:
                return True
            elif dataset == "VStarBench":
                return True
            elif dataset == "HRBench4K":
                return True
            elif "SEEDBench" in dataset:
                return True
            elif dataset in ["ZEROBench", "ZEROBench_sub", "InfoVQA_VAL", "CV-Bench-2D", "StaticEmbodiedBench",
                             "MathVista_MINI", "LogicVista", "MMMU_Pro_10c"]:
                return True
            else:
                ipdb.set_trace()  # todo see if we need custom prompt
                pass
        else:
            return super().use_custom_prompt(dataset)

    def build_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        import pandas as pd

        if self.project_name == "lpt3":
            if dataset in ["RealWorldQA", "VStarBench", "HRBench4K", "SEEDBench_IMG", "SEEDBench2", "SEEDBench2_Plus",
                           "CV-Bench-2D", "StaticEmbodiedBench"]:
                # reference: Qwen2VLPromptMixin - build_mcq_prompt
                question = line['question']
                option_names = [name for name in string.ascii_uppercase if name in line and not pd.isna(line[name])]
                if "SEEDBench" in dataset or dataset in ["CV-Bench-2D"]:
                    options_str = "\n".join(f"({option_name}) {line[option_name]}" for option_name in option_names)
                else:
                    options_str = "\n".join(f"{option_name}. {line[option_name]}" for option_name in option_names)
                prompt = f"{question}\n{options_str}".strip()

                image_path = self.dump_image(line, dataset)

                msgs = []
                if isinstance(image_path, list):
                    msgs.extend([dict(type='image', value=p) for p in image_path])
                else:
                    msgs = [dict(type='image', value=image_path)]
                msgs.append(dict(type='text', value=prompt))

            elif dataset in ["ZEROBench", "ZEROBench_sub", "InfoVQA_VAL"]:
                question = line['question'].strip()

                image_path = self.dump_image(line, dataset)

                msgs = []
                if isinstance(image_path, list):
                    msgs.extend([dict(type='image', value=p) for p in image_path])
                else:
                    msgs = [dict(type='image', value=image_path)]
                msgs.append(dict(type='text', value=question))

            elif dataset in ["CharXiv_reasoning_val"]:
                msgs = super().build_prompt(line, dataset)

                inst_to_remove = ("* Your final answer must be grounded to some text that is explicitly written and "
                                  "relevant to the question in the chart.\n    "
                                  "* If you need to answer multiple terms, separate them with commas.\n    "
                                  "* Unless specified in the question (such as answering with a letter), you are "
                                  "required to answer the full names of subplots and/or labels by default.\n")
                msgs[1]['value'] = msgs[1]['value'].split(inst_to_remove)[0].strip()

            elif dataset in ["MathVista_MINI", "LogicVista", "MMMU_Pro_10c"]:
                # just return vanilla prompt
                msgs = super().build_prompt(line, dataset)
                return msgs

            else:
                msgs = super().build_prompt(line, dataset)
                ipdb.set_trace()  # todo implement custom prompt for other dataset
                pass
                raise NotImplementedError

            if any(name in self.full_model_name for name in ["235B-v2_charts-charxiv-eng-k2-16k"]):
                msgs[1]['value'] += "\nRespond in English, and end you response with \"Final Answer: your final answer\"."

            return msgs
        else:
            return super().build_prompt(line, dataset)

    def _prepare_content_vllm(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        adapted from vlm.qwen2_vl.model
        """
        content = []
        cur_image_count = 0
        for s in inputs:
            if s['type'] == 'image':
                item = {"type": "image_url", "image_url": {"url": ensure_image_url(s['value'])}}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    self.logger.info(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels

                if cur_image_count < self.limit_mm_per_prompt:
                    content.append(item)
                    cur_image_count += 1
                else:
                    self.logger.warning(
                        f"Number of images exceeds the limit of {self.limit_mm_per_prompt}. "
                        f"Only the first {self.limit_mm_per_prompt} images will be used."
                    )
            elif s['type'] == 'video':
                ipdb.set_trace()
                raise NotImplementedError

            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
                content.append(item)

            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")

        return content

    def _parse_answer(self, generation: str, dataset: str = None) -> str | None:
        """
        Parse the answer string.
        If not found, return None.
        """
        if self.model_name in [
            "ReVisual-R1-VLLM", "MiMo-VL-7B-SFT-VLLM", "MiMo-VL-7B-RL-VLLM", "VLAA-Thinker-Qwen2.5VL-7B-VLLM",
            "NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
        ]:
            if "</think>" in generation:
                answer = generation.split("</think>")[-1].strip()
            elif len(generation) > 3000:
                # to reduce length
                answer = generation[-3000:]
            else:
                answer = generation
        elif self.model_name.startswith("lpt2-") or self.model_name in [
            "LongPerceptualThought-SFT_then_DPO",
        ] or (Path(self.model_name).exists() and self.project_name == "lpt2"):
            if candidates := re.findall(r"<answer>(.+)</answer>", generation):
                answer = candidates[-1].strip()
            elif len(generation) > 3000:
                # to reduce length
                answer = generation[-3000:]
            else:
                answer = generation
        elif Path(self.model_name).exists() and self.project_name == "vlm_rl":
            # VLM RL models
            if "</think>" in generation:
                answer = generation.split("</think>")[-1].strip()
            elif len(generation) > 3000:
                # to reduce length
                answer = "(... omitted) " + generation[-1000:]
            else:
                answer = generation
        elif Path(self.model_name).exists() and self.project_name == "lpt3" and "ablation" in self.model_name:
            # LPT3 SFT, for ablation
            if "</think>" in generation:
                answer = generation.split("</think>")[-1].strip()
            elif len(generation) > 3000:
                # to reduce length
                answer = generation[-3000:]
            else:
                answer = generation
            # else:
            #     answer = "Student answer was incomplete."
        elif Path(self.model_name).exists() and self.project_name == "lpt3":
            # Performance version
            if dataset in ["CharXiv_reasoning_val"]:
                if len(generation) > 15000:
                    # to reduce length
                    answer = "(... omitted) " + generation[-15000:]
                else:
                    answer = generation
            else:
                if "Final Answer:" in generation:
                    answer = generation.split("Final Answer:")[-1].strip()
                elif "</think>" in generation:
                    answer = generation.split("</think>")[-1].strip()
                elif len(generation) > 3000:
                    # to reduce length
                    answer = "(... omitted) " + generation[-3000:]
                else:
                    answer = generation

            # # LPT3 SFT models
            # if dataset in ["CharXiv_reasoning_val"]:
            #     if len(generation) > 15000:
            #         # to reduce length
            #         answer = "(... omitted) " + generation[-15000:]
            #     else:
            #         answer = generation
            # else:
            #     if "Final Answer:" in generation:
            #         answer = generation.split("Final Answer:")[-1].strip()
            #     elif "</think>" in generation:
            #         answer = generation.split("</think>")[-1].strip()
            #     elif len(generation) > 3000:
            #         # to reduce length
            #         answer = "(... omitted) " + generation[-3000:]
            #     else:
            #         answer = generation

            # # ablation version
            # if "</think>" in generation:
            #     answer = generation.split("</think>")[-1].strip()
            # elif len(generation) > 3000:
            #     # to reduce length
            #     answer = generation[-3000:]
            # else:
            #     answer = generation

        else:
            raise NotImplementedError

        return answer

    def get_inference_result(self, headers, payload, dataset_name):
        if self.project_name == "lpt3" and dataset_name in ["SEEDBench2_Plus"]:
            # for faster evaluation, we halve the max tokens (in the config) for some datasets
            # if generated response does not end, retry generation (all under the same token budget)
            # max_num_gen = 2
            # payload["max_tokens"] = payload["max_tokens"] // max_num_gen
            max_num_gen = 1
        else:
            max_num_gen = 1

        num_gen = 0
        ret_code, answer, response = 0, "", None
        while num_gen < max_num_gen:
            num_gen += 1
            response = requests.post(
                self.api_base,
                headers=headers, data=json.dumps(payload), timeout=self.timeout * 1.1)
            ret_code = response.status_code
            ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
            answer = self.fail_msg
            try:
                resp_struct = json.loads(response.text)
                generation = resp_struct['choices'][0]['message']['content'].strip()
                answer = self._parse_answer(generation, dataset=dataset_name)
                if answer is None:
                    answer = generation

            except Exception as err:
                generation = ""
                if self.verbose:
                    self.logger.error(f'{type(err)}: {err}')
                    self.logger.error(response.text if hasattr(response, 'text') else response)

            if "</think>" in generation:
                break

        return ret_code, answer, response



    def generate_inner(self, message, **kwargs):
        if self.config_system_prompt is not None:
            system_message = [
                {
                    "type": "text",
                    "value": self.config_system_prompt,
                }
            ]
            messages = [
                {"role": "system", "content": self._prepare_content_vllm(system_message)},
                {"role": "user", "content": self._prepare_content_vllm(message)}
            ]

        elif self.model_name.startswith("lpt2-") or self.model_name in [
            "LongPerceptualThought-SFT_then_DPO",
        ] or (Path(self.model_name).exists() and self.project_name == "lpt2"):
            system_message = [
                {
                    "type": "text",
                    "value": "A conversation between User and Assistant. The user asks a visual question, and the "
                             "Assistant solves it. The assistant first thinks about the reasoning "
                             "process in the mind and then provides the user with the answer. The "
                             "reasoning process and answer are enclosed within <think> </think> and "
                             "<answer> </answer> tags, respectively, i.e., <think> reasoning process "
                             "here </think> <answer> answer here </answer>. Please answer with the "
                             "full text of the correct option.",
                 }
            ]
            messages = [
                {"role": "system", "content": self._prepare_content_vllm(system_message)},
                {"role": "user", "content": self._prepare_content_vllm(message)}
            ]

        elif Path(self.model_name).exists() and self.project_name == "lpt3":
            messages = [
                {"role": "user", "content": self._prepare_content_vllm(message)}
            ]

        else:
            messages = [
                {"role": "user", "content": self._prepare_content_vllm(message)}
            ]

        mm_processor_kwargs = None
        for item in messages[0]['content']:
            if item['type'] == 'image' and item['min_pixels'] is not None:
                mm_processor_kwargs = {
                        "min_pixels": item['min_pixels'],
                        "max_pixels": item['max_pixels'],
                }
            break

        temperature = kwargs.pop('temperature', self.temperature)
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)

        # send API request
        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'}
        payload = dict(
            model=self.full_model_name,
            messages=messages,
            n=1,
            temperature=temperature,
            top_p=0.001,
            top_k=1,
            repetition_penalty=1.0,
            max_tokens=max_tokens,
        )
        if mm_processor_kwargs is not None:
            payload["mm_processor_kwargs"] = mm_processor_kwargs

        ret_code, answer, response = self.get_inference_result(headers, payload, kwargs["dataset"])

        return ret_code, answer, response

