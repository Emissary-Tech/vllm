# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal, Optional, Union

from fastapi import Request
from pydantic import BaseModel, Field

from vllm.entrypoints.openai.protocol import (ErrorResponse, OpenAIBaseModel,
                                             PoolingCompletionRequest, PoolingChatRequest, UsageInfo)


# Create separate classes for completion and chat requests
class ClassifyCompletionRequest(PoolingCompletionRequest):
    """Request for the classify endpoint using completion format."""
    classification_type: Literal["multiclass", "multilabel", "binary",
                                 "regression"] = "multiclass"
    chat_template: Optional[str] = Field(
        default=None,
        description=(
            "A Jinja template to use when a classification adapter requires "
            "chat template preprocessing."),
    )
    chat_template_kwargs: Optional[dict[str, Any]] = Field(
        default=None,
        description=("Additional kwargs to pass to the template renderer. "
                     "Will be accessible by the chat template."),
    )
    mm_processor_kwargs: Optional[dict[str, Any]] = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )


class ClassifyChatRequest(PoolingChatRequest):
    """Request for the classify endpoint using chat format."""
    classification_type: Literal["multiclass", "multilabel", "binary",
                                 "regression"] = "multiclass"
    add_generation_prompt: bool = Field(
        default=False,
        description=(
            "If true, add the generation prompt when rendering the chat "
            "template. Keep this aligned with the prompt format used while "
            "training the classification head."),
    )
    continue_final_message: bool = Field(
        default=False,
        description=(
            "If true, format the final chat message as open-ended. Cannot be "
            "used with add_generation_prompt."),
    )

    pass


# Define the union type for the API endpoint
ClassifyRequest = Union[ClassifyCompletionRequest, ClassifyChatRequest]


class ClassifyResponseData(BaseModel):
    """Data for a single item in a classify response."""
    index: int
    logits: Union[list[float], str]  # Either raw logits or base64 encoded
    probabilities: list[float] = []


class ClassifyResponse(OpenAIBaseModel):
    """Response for the classify endpoint."""
    id: str
    created: int
    model: str
    data: list[ClassifyResponseData]
    usage: UsageInfo
