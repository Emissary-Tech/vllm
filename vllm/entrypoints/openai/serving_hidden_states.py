# SPDX-License-Identifier: Apache-2.0

from functools import cached_property
from typing import Optional, Union

from fastapi import Request

from vllm.entrypoints.openai.protocol import (
    ErrorResponse,
    PoolingCompletionRequest,
)
from vllm.entrypoints.openai.protocol_hidden_states import (
    HiddenStatesBatchRequest,
    HiddenStatesBatchResponse,
    HiddenStatesRequest,
    HiddenStatesResponse,
)
from vllm.entrypoints.openai.serving_pooling import OpenAIServingPooling
from vllm.pooling_params import PoolingParams


class _HiddenStatesPoolingRequest(PoolingCompletionRequest):
    def to_pooling_params(self):
        return PoolingParams(return_hidden_states=True)


class OpenAIServingHiddenStates(OpenAIServingPooling):
    """Extract raw last-token features through the regular pooling scheduler.

    Prompts are already rendered by the caller. Do not apply adapter-specific
    class prompts, chat templates, normalization, or score heads here.
    """

    @cached_property
    def supports_hidden_states(self) -> bool:
        from vllm.model_executor.model_loader.utils import (
            get_model_architecture,
        )

        model_cls, _ = get_model_architecture(self.model_config)
        return getattr(
            model_cls, "supports_classification_hidden_states", False
        )

    async def create_hidden_states(
        self,
        request: Union[HiddenStatesRequest, HiddenStatesBatchRequest],
        raw_request: Optional[Request] = None,
    ) -> Union[HiddenStatesResponse, HiddenStatesBatchResponse, ErrorResponse]:
        if not self.supports_hidden_states:
            return self.create_error_response(
                "Hidden states require a converted --task=classify model, "
                "such as Qwen3ForCausalLM. "
                "Native pooling models are not supported."
            )

        batch = isinstance(request, HiddenStatesBatchRequest)
        prompts = request.prompts if batch else [request.prompt]
        # A pooling batch must be homogeneous: all text or all token ID lists.
        if any(
            isinstance(prompt, str) != isinstance(prompts[0], str)
            for prompt in prompts
        ):
            return self.create_error_response(
                "prompts must contain only strings or only token ID lists"
            )

        pooling_request = _HiddenStatesPoolingRequest(
            model=request.model,
            input=prompts,
            add_special_tokens=request.add_special_tokens,
            truncate_prompt_tokens=request.truncate_prompt_tokens,
            priority=request.priority,
        )
        response = await self.create_pooling(pooling_request, raw_request)
        if isinstance(response, ErrorResponse):
            return response
        hidden_states = [item.data for item in response.data]
        if batch:
            return HiddenStatesBatchResponse(hidden_states=hidden_states)
        return HiddenStatesResponse(hidden_states=hidden_states[0])
