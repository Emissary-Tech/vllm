# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import copy
import os
import time
from collections.abc import AsyncGenerator
from typing import Final, Literal, Optional, Union, cast

import jinja2
import numpy as np
from fastapi import Request
from typing_extensions import assert_never

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol_classify import (ClassifyCompletionRequest,
                                                      ClassifyChatRequest,
                                                      ClassifyRequest,
                                                      ClassifyResponse,
                                                      ClassifyResponseData,
                                                       ErrorResponse)
from vllm.entrypoints.openai.protocol import UsageInfo
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.logger import init_logger
from vllm.outputs import PoolingOutput, PoolingRequestOutput
from vllm.utils import merge_async_iterators
import vllm.envs as envs

logger = init_logger(__name__)


def _get_data(
    output: PoolingOutput,
    encoding_format: Literal["float", "base64"],
) -> Union[list[float], str]:
    if encoding_format == "float":
        return output.data.tolist()
    elif encoding_format == "base64":
        # Force to use float32 for base64 encoding
        # to match the OpenAI python client behavior
        pooling_bytes = np.array(output.data, dtype="float32").tobytes()
        return base64.b64encode(pooling_bytes).decode("utf-8")

    assert_never(encoding_format)


def _format_logits(
    logits: np.ndarray,
    encoding_format: Literal["float", "base64"],
) -> Union[list[float], str]:
    if encoding_format == "float":
        return logits.tolist()
    if encoding_format == "base64":
        pooling_bytes = np.array(logits, dtype="float32").tobytes()
        return base64.b64encode(pooling_bytes).decode("utf-8")
    assert_never(encoding_format)


class HFDirectClassifyRunner:

    def __init__(
        self,
        model_config: ModelConfig,
        base_model_path: str,
    ) -> None:
        # Lazy imports to avoid forcing deps unless enabled.
        import torch
        from transformers import AutoModelForSequenceClassification

        self._torch = torch
        self._auto_cls = AutoModelForSequenceClassification
        self._model_config = model_config
        self._base_model_path = base_model_path
        self._dtype = model_config.dtype
        self._trust_remote_code = model_config.trust_remote_code
        self._model = None
        self._peft_model = None
        self._num_labels: Optional[int] = None
        self._loaded_adapters: set[str] = set()
        self._lock = asyncio.Lock()

    def _find_adapter_weight_file(self, lora_path: str) -> Optional[str]:
        candidates = ("adapter_model.safetensors", "adapter_model.bin")
        for name in candidates:
            weight_path = os.path.join(lora_path, name)
            if os.path.exists(weight_path):
                return weight_path
        return None

    def _infer_num_labels(self, lora_path: str) -> int:
        weight_path = self._find_adapter_weight_file(lora_path)
        if weight_path is None:
            return int(getattr(self._model_config.hf_config, "num_labels", 1))

        if weight_path.endswith(".safetensors"):
            from safetensors import safe_open

            with safe_open(weight_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.endswith("score.weight") or key.endswith(
                            "classifier.weight"):
                        return int(f.get_tensor(key).shape[0])
        else:
            state = self._torch.load(weight_path, map_location="cpu")
            for key, tensor in state.items():
                if key.endswith("score.weight") or key.endswith(
                        "classifier.weight"):
                    return int(tensor.shape[0])

        return int(getattr(self._model_config.hf_config, "num_labels", 1))

    def _ensure_base_model(self, num_labels: int) -> None:
        if self._model is not None:
            return
        config = copy.deepcopy(self._model_config.hf_config)
        config.num_labels = num_labels
        self._model = self._auto_cls.from_pretrained(
            self._base_model_path,
            config=config,
            torch_dtype=self._dtype,
            trust_remote_code=self._trust_remote_code,
            device_map="auto",
        ).eval()
        self._num_labels = num_labels

    def _ensure_adapter(self, lora_request) -> "torch.nn.Module":
        if lora_request is None:
            if self._model is None:
                self._ensure_base_model(
                    int(getattr(self._model_config.hf_config, "num_labels",
                                1)))
            if self._peft_model is None:
                return self._model
            return self._peft_model

        import peft

        num_labels = self._infer_num_labels(lora_request.lora_path)
        if self._num_labels is not None and self._num_labels != num_labels:
            raise ValueError(
                f"LoRA adapter {lora_request.lora_name} uses num_labels={num_labels}, "
                f"but the current HF direct model is num_labels={self._num_labels}."
            )
        self._ensure_base_model(num_labels)

        if self._peft_model is None:
            self._peft_model = peft.PeftModel.from_pretrained(
                self._model,
                lora_request.lora_path,
                adapter_name=lora_request.lora_name,
                is_trainable=False,
            )
            self._loaded_adapters.add(lora_request.lora_name)
        elif lora_request.lora_name not in self._loaded_adapters:
            self._peft_model.load_adapter(lora_request.lora_path,
                                          adapter_name=lora_request.lora_name)
            self._loaded_adapters.add(lora_request.lora_name)

        self._peft_model.set_adapter(lora_request.lora_name)
        return self._peft_model

    def _run_forward(self, model, prompt_token_ids: list[list[int]]):
        device = next(model.parameters()).device
        logits_list = []
        for token_ids in prompt_token_ids:
            input_ids = self._torch.tensor([token_ids],
                                           device=device,
                                           dtype=self._torch.long)
            with self._torch.inference_mode():
                outputs = model(input_ids=input_ids, use_cache=False)
            logits = outputs.logits[0].detach().float().cpu().numpy()
            logits_list.append(logits)
        return logits_list

    async def classify(self, prompt_token_ids: list[list[int]], lora_request):
        async with self._lock:
            model = self._ensure_adapter(lora_request)
            if lora_request is None and self._peft_model is not None:
                disable_adapter = self._peft_model.disable_adapter()
            else:
                disable_adapter = None
            if disable_adapter is None:
                return await asyncio.to_thread(self._run_forward, model,
                                               prompt_token_ids)
            with disable_adapter:
                return await asyncio.to_thread(self._run_forward, model,
                                               prompt_token_ids)


class OpenAIServingClassify(OpenAIServing):

    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        models: OpenAIServingModels,
        *,
        request_logger: Optional[RequestLogger],
        chat_template: Optional[str],
        chat_template_content_format: ChatTemplateContentFormatOption,
    ) -> None:
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         models=models,
                         request_logger=request_logger)

        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format
        self._hf_direct_runner: Optional[HFDirectClassifyRunner] = None
        if envs.VLLM_HF_DIRECT_CLASSIFY:
            base_model_path = models.base_model_paths[0].model_path
            self._hf_direct_runner = HFDirectClassifyRunner(
                model_config=model_config,
                base_model_path=base_model_path,
            )
            logger.info(
                "HF direct classify is enabled. vLLM engine will be bypassed "
                "for /v1/classify requests.")

    async def create_classify(
        self,
        request: Union[ClassifyCompletionRequest, ClassifyChatRequest],
        raw_request: Optional[Request] = None,
    ) -> Union[ClassifyResponse, ErrorResponse]:
        """
        Create classification head logits for the provided input.
        This API returns the classification head logits, which can be used with
        softmax or sigmoid to get probabilities.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        encoding_format = request.encoding_format
        if request.dimensions is not None:
            return self.create_error_response(
                "dimensions is currently not supported")

        model_name = self._get_model_name(request.model)
        request_id = f"classify-{self._base_request_id(raw_request)}"
        created_time = int(time.time())

        truncate_prompt_tokens = None

        if request.truncate_prompt_tokens is not None:
            if request.truncate_prompt_tokens <= self.max_model_len:
                truncate_prompt_tokens = request.truncate_prompt_tokens
            else:
                return self.create_error_response(
                    "truncate_prompt_tokens value is "
                    "greater than max_model_len."
                    " Please, select a smaller truncation size.")

        try:
            (
                lora_request,
                prompt_adapter_request,
            ) = self._maybe_get_adapters(request)

            tokenizer = await self.engine_client.get_tokenizer(lora_request)

            if prompt_adapter_request is not None:
                raise NotImplementedError("Prompt adapter is not supported "
                                          "for classification models")

            # Process input based on request type (chat or completion)
            if isinstance(request, ClassifyChatRequest):
                (
                    _,
                    request_prompts,
                    engine_prompts,
                ) = await self._preprocess_chat(
                    request,
                    tokenizer,
                    request.messages,
                    chat_template=request.chat_template or self.chat_template,
                    chat_template_content_format=self.
                    chat_template_content_format,
                    # In classification requests, we are not generating tokens,
                    # so there is no need to append extra tokens to the input
                    add_generation_prompt=False,
                    continue_final_message=False,
                    truncate_prompt_tokens=truncate_prompt_tokens,
                    add_special_tokens=request.add_special_tokens,
                )
            else:
                (request_prompts,
                 engine_prompts) = await self._preprocess_completion(
                     request,
                     tokenizer,
                     request.input,
                     truncate_prompt_tokens=truncate_prompt_tokens,
                     add_special_tokens=request.add_special_tokens,
                 )
        except (ValueError, TypeError, jinja2.TemplateError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

        # Schedule the request and get the result generator.
        if self._hf_direct_runner is not None:
            prompt_token_ids = [
                engine_prompt["prompt_token_ids"]
                for engine_prompt in engine_prompts
            ]
            logits_list = await self._hf_direct_runner.classify(
                prompt_token_ids, lora_request)
            items = [
                ClassifyResponseData(
                    index=i,
                    logits=_format_logits(logits, encoding_format),
                ) for i, logits in enumerate(logits_list)
            ]
            num_prompt_tokens = sum(len(ids) for ids in prompt_token_ids)
            usage = UsageInfo(prompt_tokens=num_prompt_tokens,
                              total_tokens=num_prompt_tokens)
            return ClassifyResponse(id=request_id,
                                    created=created_time,
                                    model=model_name,
                                    data=items,
                                    usage=usage)

        generators: list[AsyncGenerator[PoolingRequestOutput, None]] = []
        try:
            pooling_params = request.to_pooling_params()
            # Set softmax=False to get raw logits
            # pooling_params.softmax = False

            for i, engine_prompt in enumerate(engine_prompts):
                request_id_item = f"{request_id}-{i}"

                self._log_inputs(request_id_item,
                                 request_prompts[i],
                                 params=pooling_params,
                                 lora_request=lora_request,
                                 prompt_adapter_request=prompt_adapter_request)

                trace_headers = (None if raw_request is None else await
                                 self._get_trace_headers(raw_request.headers))

                generator = self.engine_client.encode(
                    engine_prompt,
                    pooling_params,
                    request_id_item,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=request.priority,
                )

                generators.append(generator)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        result_generator = merge_async_iterators(*generators)

        num_prompts = len(engine_prompts)

        # Non-streaming response
        final_res_batch: list[Optional[PoolingRequestOutput]]
        final_res_batch = [None] * num_prompts
        try:
            async for i, res in result_generator:
                final_res_batch[i] = res

            assert all(final_res is not None for final_res in final_res_batch)

            final_res_batch_checked = cast(list[PoolingRequestOutput],
                                           final_res_batch)

            response = self.request_output_to_classify_response(
                final_res_batch_checked,
                request_id,
                created_time,
                model_name,
                encoding_format,
            )
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        return response

    def request_output_to_classify_response(
        self,
        final_res_batch: list[PoolingRequestOutput],
        request_id: str,
        created_time: int,
        model_name: str,
        encoding_format: Literal["float", "base64"],
    ) -> ClassifyResponse:
        items: list[ClassifyResponseData] = []
        num_prompt_tokens = 0

        for idx, final_res in enumerate(final_res_batch):
            item = ClassifyResponseData(
                index=idx,
                logits=_get_data(final_res.outputs, encoding_format),
            )
            prompt_token_ids = final_res.prompt_token_ids

            items.append(item)
            num_prompt_tokens += len(prompt_token_ids)

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            total_tokens=num_prompt_tokens,
        )

        return ClassifyResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            data=items,
            usage=usage,
        )
