# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from vllm.entrypoints.serve.emissary.api_router import (
    _resolve_hidden_states_lora,
    get_hidden_states,
    get_hidden_states_batch,
)
from vllm.lora.request import LoRARequest


class FakeEngineClient:
    def __init__(self, rendered_prompts):
        self.generate_calls = []
        self.renderer = MagicMock()
        self.renderer.render_chat.side_effect = [
            (None, (prompt,)) for prompt in rendered_prompts
        ]
        self.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                max_model_len=4096,
                multimodal_config=None,
            )
        )

    async def generate(
        self,
        prompt,
        sampling_params,
        request_id,
        *,
        lora_request=None,
        **kwargs,
    ):
        self.generate_calls.append(
            {
                "prompt": prompt,
                "sampling_params": sampling_params,
                "request_id": request_id,
                "lora_request": lora_request,
            }
        )
        yield SimpleNamespace(hidden_state=[float(len(self.generate_calls))])


class FakeServingModels:
    def __init__(self, lora_requests=None):
        self.lora_requests = lora_requests or {}
        self.load_lora_adapter = AsyncMock(side_effect=self._load_lora_adapter)
        self.resolve_lora = AsyncMock()

    @staticmethod
    def is_base_model(model_name):
        return model_name == "base-model"

    async def _load_lora_adapter(self, request):
        lora_request = LoRARequest(
            lora_name=request.lora_name,
            lora_int_id=11,
            lora_path=request.lora_path,
        )
        self.lora_requests[request.lora_name] = lora_request
        return "loaded"


def _make_request(body, engine, serving_models):
    state = SimpleNamespace(
        engine_client=engine,
        openai_serving_models=serving_models,
        args=SimpleNamespace(
            trust_request_chat_template=False,
            chat_template=None,
            chat_template_content_format="auto",
            default_chat_template_kwargs=None,
        ),
    )
    return SimpleNamespace(
        app=SimpleNamespace(state=state),
        json=AsyncMock(return_value=body),
    )


@pytest.mark.asyncio
async def test_resolve_hidden_states_lora_uses_loaded_model(monkeypatch):
    monkeypatch.delenv("VLLM_ALLOW_RUNTIME_LORA_UPDATING", raising=False)
    lora_request = LoRARequest(
        lora_name="classifier",
        lora_int_id=7,
        lora_path="/models/classifier",
    )
    serving_models = FakeServingModels({"classifier": lora_request})
    request = _make_request({}, FakeEngineClient([]), serving_models)

    resolved = await _resolve_hidden_states_lora(
        request,
        {"model": "classifier"},
    )

    assert resolved is lora_request
    serving_models.load_lora_adapter.assert_not_awaited()
    serving_models.resolve_lora.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_hidden_states_lora_loads_path_when_enabled(monkeypatch):
    monkeypatch.setenv("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "true")
    serving_models = FakeServingModels()
    request = _make_request({}, FakeEngineClient([]), serving_models)

    resolved = await _resolve_hidden_states_lora(
        request,
        {
            "model": "base-model",
            "lora_path": "/models/classifier",
        },
    )

    assert resolved is serving_models.lora_requests["classifier"]
    load_request = serving_models.load_lora_adapter.await_args.args[0]
    assert load_request.lora_name == "classifier"
    assert load_request.lora_path == "/models/classifier"


@pytest.mark.asyncio
async def test_resolve_hidden_states_lora_rejects_path_when_disabled(monkeypatch):
    monkeypatch.delenv("VLLM_ALLOW_RUNTIME_LORA_UPDATING", raising=False)
    serving_models = FakeServingModels()
    request = _make_request({}, FakeEngineClient([]), serving_models)

    with pytest.raises(HTTPException, match="VLLM_ALLOW_RUNTIME_LORA_UPDATING"):
        await _resolve_hidden_states_lora(
            request,
            {
                "lora_name": "classifier",
                "lora_path": "/models/classifier",
            },
        )

    serving_models.load_lora_adapter.assert_not_awaited()


@pytest.mark.asyncio
async def test_hidden_states_passes_lora_with_rendered_multimodal_prompt(monkeypatch):
    monkeypatch.delenv("VLLM_ALLOW_RUNTIME_LORA_UPDATING", raising=False)
    rendered_prompt = {
        "prompt_token_ids": [1, 2, 3],
        "multi_modal_data": {"image": object()},
    }
    engine = FakeEngineClient([rendered_prompt])
    lora_request = LoRARequest(
        lora_name="classifier",
        lora_int_id=7,
        lora_path="/models/classifier",
    )
    serving_models = FakeServingModels({"classifier": lora_request})
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}},
                {"type": "text", "text": "Classify this image."},
            ],
        }
    ]
    request = _make_request(
        {
            "model": "classifier",
            "messages": messages,
            "add_generation_prompt": False,
        },
        engine,
        serving_models,
    )

    response = await get_hidden_states(request)

    assert json.loads(response.body) == {"hidden_states": [1.0]}
    assert len(engine.generate_calls) == 1
    assert engine.generate_calls[0]["prompt"] is rendered_prompt
    assert engine.generate_calls[0]["lora_request"] is lora_request
    render_args = engine.renderer.render_chat.call_args.args
    assert render_args[0] == [messages]
    assert render_args[1].chat_template_kwargs["add_generation_prompt"] is False


@pytest.mark.asyncio
async def test_hidden_states_batch_shares_lora_across_rendered_prompts(monkeypatch):
    monkeypatch.delenv("VLLM_ALLOW_RUNTIME_LORA_UPDATING", raising=False)
    rendered_prompts = [
        {
            "prompt_token_ids": [1, 2],
            "multi_modal_data": {"image": object()},
        },
        {
            "prompt_token_ids": [3, 4],
            "multi_modal_data": {"image": object()},
        },
    ]
    engine = FakeEngineClient(rendered_prompts)
    lora_request = LoRARequest(
        lora_name="classifier",
        lora_int_id=7,
        lora_path="/models/classifier",
    )
    serving_models = FakeServingModels({"classifier": lora_request})
    request = _make_request(
        {
            "model": "classifier",
            "add_generation_prompt": False,
            "prompts": [
                {"messages": [{"role": "user", "content": "first"}]},
                {"messages": [{"role": "user", "content": "second"}]},
            ],
        },
        engine,
        serving_models,
    )

    response = await get_hidden_states_batch(request)

    assert json.loads(response.body) == {"hidden_states": [[1.0], [2.0]]}
    assert [call["prompt"] for call in engine.generate_calls] == rendered_prompts
    assert all(call["lora_request"] is lora_request for call in engine.generate_calls)
    serving_models.resolve_lora.assert_not_awaited()
