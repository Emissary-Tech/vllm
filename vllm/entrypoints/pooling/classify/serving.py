# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
from typing import Literal, TypeAlias

import numpy as np
from fastapi.responses import JSONResponse

from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import ChatTemplateConfig
from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.entrypoints.pooling.base.serving import PoolingServing
from vllm.entrypoints.pooling.typing import PoolingServeContext
from vllm.logger import init_logger
from vllm.renderers import BaseRenderer

from .io_processor import ClassifyIOProcessor
from .protocol import (
    ClassificationData,
    ClassificationRequest,
    ClassificationResponse,
)

logger = init_logger(__name__)


ClassificationServeContext: TypeAlias = PoolingServeContext[ClassificationRequest]


def _get_logits_data(
    pooled_data,
    encoding_format: Literal["float", "base64"],
) -> list[float] | str:
    if encoding_format == "float":
        return pooled_data.tolist()
    if encoding_format == "base64":
        logits_bytes = np.array(pooled_data, dtype="float32").tobytes()
        return base64.b64encode(logits_bytes).decode("utf-8")
    raise ValueError(f"Unsupported classify encoding_format: {encoding_format}")


class ServingClassification(PoolingServing):
    request_id_prefix = "classify"

    def init_io_processor(
        self,
        model_config: ModelConfig,
        renderer: BaseRenderer,
        chat_template_config: ChatTemplateConfig,
    ) -> ClassifyIOProcessor:
        return ClassifyIOProcessor(
            model_config=model_config,
            renderer=renderer,
            chat_template_config=chat_template_config,
        )

    async def _build_response(
        self,
        ctx: ClassificationServeContext,
    ) -> JSONResponse:
        if ctx.request.dimensions is not None:
            raise ValueError("dimensions is currently not supported")

        num_prompt_tokens = 0
        items: list[ClassificationData] = []
        encoding_format = ctx.request.encoding_format
        for idx, final_res in enumerate(ctx.final_res_batch):
            item = ClassificationData(
                index=idx,
                logits=_get_logits_data(final_res.outputs.data, encoding_format),
            )

            items.append(item)
            prompt_token_ids = final_res.prompt_token_ids
            num_prompt_tokens += len(prompt_token_ids)

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            total_tokens=num_prompt_tokens,
        )

        response = ClassificationResponse(
            id=ctx.request_id,
            created=ctx.created_time,
            model=ctx.request.model or ctx.model_name,
            data=items,
            usage=usage,
        )

        return JSONResponse(content=response.model_dump())
