import json
import re
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
        elif model_name.startswith("lpt2-"):
            self.full_model_name = f"Jaehun/{self.model_name}"
        elif Path(model_name).exists():
            # local checkpoint
            if "vlm_rl" in model_name:
                self.project_name = "vlm_rl"
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

    def _parse_answer(self, generation: str) -> str | None:
        """
        Parse the answer string.
        If not found, return None.
        """
        if self.model_name in [
            "ReVisual-R1-VLLM", "MiMo-VL-7B-SFT-VLLM", "MiMo-VL-7B-RL-VLLM", "VLAA-Thinker-Qwen2.5VL-7B-VLLM",
        ]:
            if "</think>" in generation:
                answer = generation.split("</think>")[-1].strip()
            elif len(generation) > 3000:
                # to reduce length
                answer = generation[-3000:]
            else:
                answer = ""
        elif self.model_name.startswith("lpt2-"):
            if candidates := re.findall(r"<answer>(.+)</answer>", generation):
                answer = candidates[-1].strip()
            elif len(generation) > 3000:
                # to reduce length
                answer = generation[-3000:]
            else:
                answer = ""
        elif Path(self.model_name).exists() and self.project_name == "vlm_rl":
            # VLM RL models
            if "</think>" in generation:
                answer = generation.split("</think>")[-1].strip()
            elif len(generation) > 3000:
                # to reduce length
                answer = generation[-3000:]
            else:
                answer = ""
        else:
            raise NotImplementedError

        return answer

    def generate_inner(self, message, **kwargs):
        if self.model_name.startswith("lpt2-"):
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

        elif self.config_system_prompt is not None:
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

        response = requests.post(
            self.api_base,
            headers=headers, data=json.dumps(payload), timeout=self.timeout * 1.1)
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
        answer = self.fail_msg
        try:
            resp_struct = json.loads(response.text)
            generation = resp_struct['choices'][0]['message']['content'].strip()
            answer = self._parse_answer(generation)
            if answer is None:
                answer = generation
        except Exception as err:
            if self.verbose:
                self.logger.error(f'{type(err)}: {err}')
                self.logger.error(response.text if hasattr(response, 'text') else response)

        return ret_code, answer, response

