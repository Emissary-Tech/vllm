# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import os
import uuid
from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from typing_extensions import Never

import vllm.envs as envs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import load_chat_template
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.models.api_router import models
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.serve.lora.protocol import LoadLoRAAdapterRequest
from vllm.inputs.data import ProcessorInputs, PromptType
from vllm.lora.request import LoRARequest
from vllm.renderers import ChatParams, TokenizeParams, merge_kwargs
from vllm.sampling_params import RequestOutputKind, SamplingParams

router = APIRouter()


def attach_router(app: FastAPI) -> None:
    app.include_router(router)


def _engine_client(raw_request: Request) -> EngineClient:
    client = getattr(raw_request.app.state, "engine_client", None)
    if client is None:
        raise HTTPException(
            status_code=HTTPStatus.NOT_IMPLEMENTED.value,
            detail="This endpoint requires a running vLLM engine.",
        )
    return client


def _hidden_state_sampling_params() -> SamplingParams:
    return SamplingParams(
        max_tokens=1,
        temperature=0.0,
        detokenize=False,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )


def _validate_token_ids(token_ids: Any) -> list[int]:
    if not isinstance(token_ids, list) or not token_ids:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="'token_ids' must be a non-empty list.",
        )
    try:
        return [int(token_id) for token_id in token_ids]
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="'token_ids' must contain integers.",
        ) from exc


def _prompt_extras(body: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key in (
            "multi_modal_data",
            "mm_processor_kwargs",
            "multi_modal_uuids",
            "cache_salt",
        )
        if (value := body.get(key)) is not None
    }


EnginePrompt = PromptType | ProcessorInputs


def _optional_body_str(body: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = body.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=f"'{key}' must be a non-empty string when provided.",
            )
        return value.strip()
    return None


def _default_lora_name_from_path(lora_path: str) -> str:
    path = lora_path.rstrip("/").rstrip("\\")
    return os.path.basename(path) or f"hidden-states-lora-{uuid.uuid4().hex}"


def _raise_lora_error(error: ErrorResponse) -> Never:
    raise HTTPException(
        status_code=error.error.code,
        detail=error.error.message,
    )


async def _resolve_hidden_states_lora(
    raw_request: Request,
    body: dict[str, Any],
) -> LoRARequest | None:
    """Resolve the adapter requested by a hidden-state endpoint."""
    lora_path = _optional_body_str(body, "lora_path")
    lora_name = _optional_body_str(body, "lora_name", "lora", "adapter_name")
    model_name = _optional_body_str(body, "model")
    serving_models = models(raw_request)

    if lora_path is not None:
        resolved_name = lora_name or model_name
        if not resolved_name or serving_models.is_base_model(resolved_name):
            resolved_name = _default_lora_name_from_path(lora_path)

        loaded_lora = serving_models.lora_requests.get(resolved_name)
        if loaded_lora is not None:
            if loaded_lora.lora_path != lora_path:
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST.value,
                    detail=(
                        f"LoRA adapter '{resolved_name}' is already loaded from "
                        f"'{loaded_lora.lora_path}', not '{lora_path}'. Use a "
                        "different lora_name."
                    ),
                )
            return loaded_lora

        if not envs.VLLM_ALLOW_RUNTIME_LORA_UPDATING:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=(
                    "Loading a LoRA from 'lora_path' requires "
                    "VLLM_ALLOW_RUNTIME_LORA_UPDATING=True."
                ),
            )

        load_result = await serving_models.load_lora_adapter(
            LoadLoRAAdapterRequest(
                lora_name=resolved_name,
                lora_path=lora_path,
            )
        )
        if isinstance(load_result, ErrorResponse):
            # Another request may have loaded the same adapter while this one
            # was waiting for the per-name lock.
            loaded_lora = serving_models.lora_requests.get(resolved_name)
            if loaded_lora is not None and loaded_lora.lora_path == lora_path:
                return loaded_lora
            _raise_lora_error(load_result)
        return serving_models.lora_requests[resolved_name]

    resolved_name = lora_name or model_name
    if not resolved_name or serving_models.is_base_model(resolved_name):
        return None

    loaded_lora = serving_models.lora_requests.get(resolved_name)
    if loaded_lora is not None:
        return loaded_lora

    if not envs.VLLM_ALLOW_RUNTIME_LORA_UPDATING:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND.value,
            detail=f"The model '{resolved_name}' does not exist.",
        )

    resolved_lora = await serving_models.resolve_lora(resolved_name)
    if isinstance(resolved_lora, ErrorResponse):
        _raise_lora_error(resolved_lora)
    return resolved_lora


def _render_messages(raw_request: Request, body: dict[str, Any]) -> EnginePrompt:
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="'messages' must be a list.",
        )

    args = raw_request.app.state.args
    has_request_template = (
        body.get("chat_template") is not None
        or (body.get("chat_template_kwargs") or {}).get("chat_template") is not None
    )
    if not args.trust_request_chat_template and has_request_template:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Request chat templates require --trust-request-chat-template.",
        )

    client = _engine_client(raw_request)
    model_config = client.vllm_config.model_config
    mm_config = model_config.multimodal_config
    request_chat_kwargs = merge_kwargs(
        body.get("chat_template_kwargs"),
        {
            "add_generation_prompt": body.get("add_generation_prompt", True),
            "continue_final_message": body.get("continue_final_message", False),
        },
    )
    chat_params = ChatParams(
        chat_template=(
            body.get("chat_template") or load_chat_template(args.chat_template)
        ),
        chat_template_content_format=args.chat_template_content_format,
        chat_template_kwargs=request_chat_kwargs,
        media_io_kwargs=body.get("media_io_kwargs"),
        mm_processor_kwargs=body.get("mm_processor_kwargs"),
    ).with_defaults(
        default_chat_template_kwargs=args.default_chat_template_kwargs,
        default_media_io_kwargs=(mm_config.media_io_kwargs if mm_config else None),
    )
    tok_params = TokenizeParams(
        max_total_tokens=model_config.max_model_len,
        max_output_tokens=1,
        truncate_prompt_tokens=body.get("truncate_prompt_tokens"),
        truncation_side=body.get("truncation_side"),
        add_special_tokens=body.get("add_special_tokens", False),
        max_total_tokens_param="max_model_len",
        max_output_tokens_param="max_tokens",
    )

    _, (engine_prompt,) = client.renderer.render_chat(
        [messages], chat_params, tok_params
    )
    return engine_prompt


def _prepare_engine_prompt(raw_request: Request, body: dict[str, Any]) -> EnginePrompt:
    if "messages" in body:
        return _render_messages(raw_request, body)

    if "prompt" not in body and "input" in body:
        body = {**body, "prompt": body["input"]}

    if "prompt" not in body:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Provide 'prompt' or 'messages'.",
        )

    prompt = body["prompt"]
    if isinstance(prompt, dict):
        if "messages" in prompt:
            return _render_messages(raw_request, {**body, **prompt})
        return prompt

    extras = _prompt_extras(body)
    if extras:
        if isinstance(prompt, str):
            return {"prompt": prompt, **extras}
        if isinstance(prompt, list) and all(isinstance(t, int) for t in prompt):
            return {"prompt_token_ids": prompt, **extras}

    if isinstance(prompt, (str, list)):
        return prompt

    raise HTTPException(
        status_code=HTTPStatus.BAD_REQUEST.value,
        detail="'prompt' must be a string, token id list, or prompt object.",
    )


async def _run_hidden_state_request(
    client: EngineClient,
    prompt: EnginePrompt,
    lora_request: LoRARequest | None = None,
) -> list[float]:
    request_id = f"hs-{uuid.uuid4().hex}"
    final_output = None
    async for output in client.generate(
        prompt,
        _hidden_state_sampling_params(),
        request_id,
        lora_request=lora_request,
    ):
        final_output = output

    hidden_state = getattr(final_output, "hidden_state", None)
    if hidden_state is None:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=(
                "Hidden states not available. Start the server with "
                "--return-hidden-states."
            ),
        )
    return hidden_state


@router.post("/v1/lm_head_weights", dependencies=[Depends(validate_json_request)])
async def get_lm_head_weights(raw_request: Request):
    body = await raw_request.json()
    token_ids = _validate_token_ids(body.get("token_ids"))
    client = _engine_client(raw_request)

    results = await client.collective_rpc(
        "get_lm_head_weights",
        args=(token_ids,),
    )
    weights = next((result for result in results if result is not None), None)
    if weights is None:
        raise HTTPException(
            status_code=HTTPStatus.NOT_IMPLEMENTED.value,
            detail="The loaded model does not expose lm_head weights.",
        )

    return JSONResponse(content={"weights": weights})


@router.post("/v1/hidden_states", dependencies=[Depends(validate_json_request)])
async def get_hidden_states(raw_request: Request):
    body = await raw_request.json()
    client = _engine_client(raw_request)
    prompt = _prepare_engine_prompt(raw_request, body)
    lora_request = await _resolve_hidden_states_lora(raw_request, body)
    hidden_state = await _run_hidden_state_request(client, prompt, lora_request)
    return JSONResponse(content={"hidden_states": hidden_state})


@router.post("/v1/hidden_states_batch", dependencies=[Depends(validate_json_request)])
async def get_hidden_states_batch(raw_request: Request):
    body = await raw_request.json()
    raw_prompts = body.get("prompts")
    if not isinstance(raw_prompts, list):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="'prompts' must be a list.",
        )
    if not raw_prompts:
        return JSONResponse(content={"hidden_states": []})

    client = _engine_client(raw_request)
    lora_request = await _resolve_hidden_states_lora(raw_request, body)
    prompts = []
    for item in raw_prompts:
        item_body = (
            {**body, **item}
            if isinstance(item, dict)
            else {
                **body,
                "prompt": item,
            }
        )
        item_body.pop("prompts", None)
        prompts.append(_prepare_engine_prompt(raw_request, item_body))
    hidden_states = await asyncio.gather(
        *[_run_hidden_state_request(client, prompt, lora_request) for prompt in prompts]
    )
    return JSONResponse(content={"hidden_states": hidden_states})
