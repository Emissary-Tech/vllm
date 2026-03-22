# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from typing import Literal, TypeAlias

from pydantic import Field

from vllm import PoolingParams
from vllm.config import ModelConfig
from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel, UsageInfo
from vllm.entrypoints.pooling.base.protocol import (
    ChatRequestMixin,
    ClassifyRequestMixin,
    CompletionRequestMixin,
    PoolingBasicRequestMixin,
)
from vllm.logger import init_logger
from vllm.renderers import TokenizeParams
from vllm.utils import random_uuid

logger = init_logger(__name__)


class LegacyClassifyRequestMixin(OpenAIBaseModel):
    classification_type: Literal["multiclass", "multilabel"] | None = None
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = None


class ClassificationCompletionRequest(
    PoolingBasicRequestMixin,
    CompletionRequestMixin,
    ClassifyRequestMixin,
    LegacyClassifyRequestMixin,
):
    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        encoder_config = model_config.encoder_config or {}

        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            max_output_tokens=0,
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            truncation_side=self.truncation_side,
            do_lower_case=encoder_config.get("do_lower_case", False),
            add_special_tokens=self.add_special_tokens,
            max_total_tokens_param="max_model_len",
        )

    def to_pooling_params(self):
        return PoolingParams(
            task="classify",
            use_activation=False if self.use_activation is None else self.use_activation,
        )


class ClassificationChatRequest(
    PoolingBasicRequestMixin,
    ChatRequestMixin,
    ClassifyRequestMixin,
    LegacyClassifyRequestMixin,
):
    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        encoder_config = model_config.encoder_config or {}

        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            max_output_tokens=0,
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            truncation_side=self.truncation_side,
            do_lower_case=encoder_config.get("do_lower_case", False),
            add_special_tokens=self.add_special_tokens,
            max_total_tokens_param="max_model_len",
        )

    def to_pooling_params(self):
        return PoolingParams(
            task="classify",
            use_activation=False if self.use_activation is None else self.use_activation,
        )


ClassificationRequest: TypeAlias = (
    ClassificationCompletionRequest | ClassificationChatRequest
)


class ClassificationData(OpenAIBaseModel):
    index: int
    logits: list[float] | str
    probabilities: list[float] = Field(default_factory=list)


class ClassificationResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"classify-{random_uuid()}")
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    data: list[ClassificationData]
    usage: UsageInfo
