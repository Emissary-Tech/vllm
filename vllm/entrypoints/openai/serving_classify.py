# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import json
import os
import time
from collections.abc import AsyncGenerator
from functools import lru_cache
from typing import Any, Final, Literal, Optional, Union, cast

import jinja2
import numpy as np
import torch
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
from vllm.lora.request import LoRARequest
from vllm.outputs import PoolingOutput, PoolingRequestOutput
from vllm.utils import merge_async_iterators

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


@lru_cache(maxsize=128)
def _load_json_file(path: str) -> dict[str, Any]:
    try:
        with open(path) as f:
            return json.load(f)
    except OSError:
        return {}


@lru_cache(maxsize=128)
def _load_adapter_metadata(adapter_path: str) -> dict[str, Any]:
    adapter_config = _load_json_file(
        os.path.join(adapter_path, "adapter_config.json"))
    task_details = _load_json_file(
        os.path.join(adapter_path, "task_details.json"))
    metadata = dict(task_details)
    metadata.update(adapter_config)
    return metadata


def _get_adapter_config(
    lora_request: Optional[LoRARequest],
) -> dict[str, Any]:
    if lora_request is None:
        return {}
    return _load_adapter_metadata(lora_request.lora_path)


def _is_zero_shot_head(adapter_config: dict[str, Any]) -> bool:
    target_modules = adapter_config.get("target_modules")
    modules_to_save = adapter_config.get("modules_to_save") or []
    if isinstance(modules_to_save, str):
        modules_to_save = [modules_to_save]
    saved_heads = {"score", "classifier"}.intersection(modules_to_save)
    return target_modules == [] and bool(saved_heads)


def _get_response_activation(
    request: ClassifyRequest,
    adapter_config: dict[str, Any],
) -> str:
    del adapter_config
    if request.classification_type == "regression":
        return "identity"
    if request.classification_type in ("multilabel", "binary"):
        return "sigmoid"
    return "softmax"


def _get_probabilities(
    output: PoolingOutput,
    activation: str,
) -> list[float]:
    if activation in ("identity", "regression", "none"):
        return []

    logits = output.data.to(dtype=torch.float32)
    if activation in ("softmax", "multiclass"):
        return torch.softmax(logits, dim=-1).tolist()
    if activation in ("sigmoid", "multilabel"):
        return torch.sigmoid(logits).tolist()

    return []


def _should_apply_chat_template(adapter_config: dict[str, Any]) -> bool:
    return _is_zero_shot_head(adapter_config)


def _completion_input_to_chat_messages(
    input_data: Union[list[int], list[list[int]], str, list[str]],
) -> Optional[list[list[dict[str, str]]]]:
    if isinstance(input_data, str):
        return [[{"role": "user", "content": input_data}]]
    if (isinstance(input_data, list)
            and all(isinstance(item, str) for item in input_data)):
        return [[{"role": "user", "content": item}] for item in input_data]
    return None


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
            adapter_config = _get_adapter_config(lora_request)
            response_activation = _get_response_activation(
                request, adapter_config)

            if prompt_adapter_request is not None:
                raise NotImplementedError("Prompt adapter is not supported "
                                          "for classification models")

            # Process input based on request type (chat or completion)
            if (isinstance(request, ClassifyChatRequest)
                    or _should_apply_chat_template(adapter_config)):
                if isinstance(request, ClassifyChatRequest):
                    message_batches = [request.messages]
                else:
                    message_batches = _completion_input_to_chat_messages(
                        request.input)
                    if message_batches is None:
                        return self.create_error_response(
                            "This classification adapter requires chat "
                            "template preprocessing, so input must be a "
                            "string or list of strings.")

                request_prompts = []
                engine_prompts = []
                is_zero_shot_head = _is_zero_shot_head(adapter_config)
                chat_template_kwargs = {}
                if is_zero_shot_head:
                    chat_template_kwargs.setdefault("enable_thinking", False)
                chat_template_kwargs.update(
                    getattr(request, "chat_template_kwargs", None) or {})
                add_generation_prompt = (
                    True if is_zero_shot_head else getattr(
                        request, "add_generation_prompt", False))
                continue_final_message = getattr(
                    request, "continue_final_message", False)

                for messages in message_batches:
                    (
                        _,
                        request_prompts_i,
                        engine_prompts_i,
                    ) = await self._preprocess_chat(
                        request,
                        tokenizer,
                        messages,
                        chat_template=getattr(request, "chat_template", None)
                        or self.chat_template,
                        chat_template_content_format=self.
                        chat_template_content_format,
                        add_generation_prompt=add_generation_prompt,
                        continue_final_message=continue_final_message,
                        chat_template_kwargs=chat_template_kwargs,
                        truncate_prompt_tokens=truncate_prompt_tokens,
                        add_special_tokens=request.add_special_tokens,
                    )
                    request_prompts.extend(request_prompts_i)
                    engine_prompts.extend(engine_prompts_i)
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
                response_activation,
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
        response_activation: str,
    ) -> ClassifyResponse:
        items: list[ClassifyResponseData] = []
        num_prompt_tokens = 0

        for idx, final_res in enumerate(final_res_batch):
            item = ClassifyResponseData(
                index=idx,
                logits=_get_data(final_res.outputs, encoding_format),
                probabilities=_get_probabilities(final_res.outputs,
                                                 response_activation),
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
