# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

TokenId = Annotated[StrictInt, Field(ge=0)]
HiddenStatesPrompt = Union[
    Annotated[StrictStr, Field(min_length=1)],
    Annotated[list[TokenId], Field(min_length=1)],
]


class HiddenStatesRequestBase(BaseModel):
    # Reject legacy path/id fields instead of silently running the base model.
    model_config = ConfigDict(extra="forbid")
    model: Optional[str] = None
    add_special_tokens: bool = True
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None
    priority: int = 0


class HiddenStatesRequest(HiddenStatesRequestBase):
    prompt: HiddenStatesPrompt


class HiddenStatesBatchRequest(HiddenStatesRequestBase):
    prompts: Annotated[list[HiddenStatesPrompt], Field(min_length=1)]


class HiddenStatesResponse(BaseModel):
    hidden_states: list[float]


class HiddenStatesBatchResponse(BaseModel):
    hidden_states: list[list[float]]
