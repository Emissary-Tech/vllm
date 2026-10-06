# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts; run with --confcutdir=tests/classification_cpu.

Load the dependency-light production modules directly so these tests do not
require importing vLLM's CUDA engine (or tests/conftest.py).
"""

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Optional, Union

import msgspec
import pytest
import torch
import torch.nn.functional as F
from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).resolve().parents[2]


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


heads = load_source(
    "cpu_classification_head",
    "vllm/model_executor/layers/classification_head.py",
)
protocol = load_source(
    "cpu_hidden_protocol", "vllm/entrypoints/openai/protocol_hidden_states.py"
)
params = load_source("cpu_pooling_params", "vllm/pooling_params.py")
WEIGHT = "base_model.model.score.weight"
BIAS = "base_model.model.score.bias"


def test_mixed_batch_routes_each_head_and_bypasses_raw_rows():
    h = torch.tensor(
        [
            [3.0, 4.0],
            [5.0, 12.0],
            [8.0, 15.0],
            [7.0, 24.0],
            [20.0, 21.0],
            [9.0, 40.0],
        ]
    )
    linear = heads.ClassificationHead(torch.eye(2), torch.tensor([1.0, -1.0]))
    regression = heads.ClassificationHead(
        torch.tensor([[2.0, -3.0]]),
        torch.tensor([0.5]),
        torch.tensor([1.0, 2.0]),
    )
    cosine = heads.ClassificationHead(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), use_cosine=True
    )
    adapters = {11: linear, 22: regression, 33: cosine}
    lookups = []

    def get_head(adapter_id):
        lookups.append(adapter_id)
        return adapters[adapter_id]

    base = lambda x: x.sum(-1, keepdim=True)
    outputs = heads.classify_rows(
        h,
        [11, 22, 33, 11, 0, 999],
        [False, False, False, False, False, True],
        get_head,
        base,
    )
    assert [len(row) for row in outputs] == [2, 1, 3, 2, 1, 2]
    expected = [
        linear(h[0]),
        regression(h[1]),
        cosine(h[2]),
        linear(h[3]),
        base(h[4]),
        h[5],
    ]
    for actual, wanted in zip(outputs, expected):
        torch.testing.assert_close(actual, wanted)
    assert lookups == [11, 22, 33]
    # No normalization or regression centering leaked into the raw output.
    torch.testing.assert_close(outputs[-1], torch.tensor([9.0, 40.0]))


def test_alternating_adapters_and_base_do_not_change_shared_score():
    weight = torch.tensor([[1.0, 2.0]])
    original = weight.clone()
    adapters = {
        1: heads.ClassificationHead(torch.eye(2)),
        2: heads.ClassificationHead(-torch.eye(2)),
    }
    h = torch.tensor([[2.0, 3.0]])
    base = lambda x: F.linear(x, weight)
    for adapter_id in [1, 2, 0, 1, 0, 2]:
        row = heads.classify_rows(
            h, [adapter_id], [False], adapters.__getitem__, base
        )[0]
        expected = adapters[adapter_id](h)[0] if adapter_id else base(h)[0]
        torch.testing.assert_close(row, expected)
    torch.testing.assert_close(weight, original)


def test_missing_head_cannot_fall_back_to_another_adapter():
    with pytest.raises(KeyError):
        heads.classify_rows(
            torch.ones(1, 2), [99], [False], {}.__getitem__, lambda x: x
        )


def test_regression_matches_playground_before_clipping():
    v_base = torch.tensor([0.3, -0.2, 0.1])
    low = F.normalize(torch.tensor([-1.0, 0.5, 0.2]) - v_base, dim=0)
    high = F.normalize(torch.tensor([1.0, 0.2, -0.4]) - v_base, dim=0)
    direction = high - low
    weight = (10 * direction / direction.dot(direction)).unsqueeze(0)
    bias = -10 * low.dot(direction) / direction.dot(direction)
    head = heads.ClassificationHead.from_tensors(
        {
            WEIGHT: weight,
            BIAS: bias.unsqueeze(0),
            heads.REGRESSION_V_BASE_KEY: v_base,
        },
        head_only=True,
    )
    h = torch.tensor([[1.0, 0.2, -0.4], [-1.0, 0.5, 0.2]])
    torch.testing.assert_close(
        head(h).flatten(), torch.tensor([10.0, 0.0]), atol=1e-6, rtol=1e-5
    )
    assert not head.use_cosine


def test_binary_head_and_lora_head_keep_linear_scoring():
    for head_only, weight in [
        (True, torch.tensor([[2.0, 3.0]])),
        (False, torch.eye(2)),
    ]:
        head = heads.ClassificationHead.from_tensors(
            {WEIGHT: weight}, head_only=head_only
        )
        assert not head.use_cosine
        h = torch.tensor([[3.0, 4.0]])
        torch.testing.assert_close(head(h), F.linear(h, weight))


def test_head_only_multiclass_matches_cosine_temperature():
    weight = torch.tensor([[1.0, 2.0], [-2.0, 1.0]])
    head = heads.ClassificationHead.from_tensors(
        {WEIGHT: weight}, head_only=True
    )
    h = torch.tensor([[3.0, 4.0]])
    torch.testing.assert_close(
        head(h),
        40 * F.linear(F.normalize(h, dim=-1), F.normalize(weight, dim=-1)),
    )


@pytest.mark.parametrize(
    "tensors",
    [
        {WEIGHT: torch.ones(2)},
        {WEIGHT: torch.ones(2, 3), BIAS: torch.ones(1)},
        {WEIGHT: torch.ones(1, 3), heads.REGRESSION_V_BASE_KEY: torch.ones(2)},
        {BIAS: torch.ones(1)},
        {WEIGHT: torch.ones(2, 3), heads.REGRESSION_V_BASE_KEY: torch.ones(3)},
    ],
)
def test_invalid_head_shapes_are_rejected(tensors):
    with pytest.raises(ValueError):
        heads.ClassificationHead.from_tensors(tensors, head_only=True)


def test_regression_center_keeps_float32_when_weights_are_cast():
    head = heads.ClassificationHead(torch.ones(1, 2), v_base=torch.ones(2))
    cast = head.to(torch.device("cpu"), torch.bfloat16)
    assert cast.weight.dtype == torch.bfloat16
    assert cast.v_base.dtype == torch.float32
    assert head.weight.dtype == torch.float32


def test_pooling_hidden_flag_survives_clone_and_rpc_serialization():
    source = params.PoolingParams(return_hidden_states=True)
    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(source), type=params.PoolingParams
    )
    assert decoded.clone().return_hidden_states
    assert not params.PoolingParams().return_hidden_states


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": ""},
        {"prompt": []},
        {"prompt": [-1]},
        {"prompt": [True]},
        {"prompt": [1.5]},
        {"prompt": "test", "lora_path": "/unregistered/adapter"},
    ],
)
def test_invalid_hidden_requests_cannot_silently_run_base_model(payload):
    with pytest.raises(ValidationError):
        protocol.HiddenStatesRequest.model_validate(payload)


def test_hidden_request_accepts_registered_model_and_token_ids():
    request = protocol.HiddenStatesRequest(model="adapter-a", prompt=[1, 2, 3])
    assert request.model == "adapter-a"
    assert request.prompt == [1, 2, 3]
    with pytest.raises(ValidationError):
        protocol.HiddenStatesBatchRequest(prompts=[])


@pytest.fixture
def hidden_service(monkeypatch):
    # Mock the unchanged pooling transport; exercise the production hidden
    # service's request conversion and response handling without a GPU engine.
    class ErrorResponse(BaseModel):
        message: str
        code: int = 400

    class PoolingRequest(BaseModel):
        model: Optional[str] = None
        input: Union[list[str], list[list[int]]]
        add_special_tokens: bool
        truncate_prompt_tokens: Optional[int] = None
        priority: int

    class PoolingService:
        def create_error_response(self, message):
            return ErrorResponse(message=message)

    dependencies = {
        "vllm.entrypoints.openai.protocol": {
            "ErrorResponse": ErrorResponse,
            "PoolingCompletionRequest": PoolingRequest,
        },
        "vllm.entrypoints.openai.serving_pooling": {
            "OpenAIServingPooling": PoolingService,
        },
    }
    for name, attributes in dependencies.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(
        sys.modules, "vllm.entrypoints.openai.protocol_hidden_states", protocol
    )
    monkeypatch.setitem(sys.modules, "vllm.pooling_params", params)
    serving = load_source(
        "cpu_hidden_serving", "vllm/entrypoints/openai/serving_hidden_states.py"
    )
    service = serving.OpenAIServingHiddenStates()
    service.supports_hidden_states = True
    service.calls = []

    async def create_pooling(request, raw_request):
        service.calls.append((request, raw_request))
        if request.model == "unknown":
            return ErrorResponse(message="Model not found", code=404)
        return SimpleNamespace(
            data=[SimpleNamespace(data=[30.0, 40.0]) for _ in request.input]
        )

    service.create_pooling = create_pooling
    return service


def test_service_preserves_prompt_model_and_raw_pooling_flag(hidden_service):
    request = protocol.HiddenStatesRequest(
        model="adapter-a",
        prompt="<rendered prompt>",
        add_special_tokens=False,
        truncate_prompt_tokens=64,
        priority=3,
    )
    response = asyncio.run(hidden_service.create_hidden_states(request))
    assert response.hidden_states == [30.0, 40.0]
    sent, _ = hidden_service.calls[0]
    assert sent.model == "adapter-a"
    assert sent.input == ["<rendered prompt>"]
    assert not sent.add_special_tokens
    assert sent.truncate_prompt_tokens == 64
    assert sent.priority == 3
    assert sent.to_pooling_params().return_hidden_states


def test_service_batch_keeps_shape_and_propagates_model_errors(hidden_service):
    request = protocol.HiddenStatesBatchRequest(prompts=[[1, 2], [3, 4]])
    response = asyncio.run(hidden_service.create_hidden_states(request))
    assert response.hidden_states == [[30.0, 40.0], [30.0, 40.0]]
    error = asyncio.run(
        hidden_service.create_hidden_states(
            protocol.HiddenStatesRequest(model="unknown", prompt="text")
        )
    )
    assert error.code == 404


def test_service_rejects_unsupported_models_and_mixed_batches(hidden_service):
    mixed = protocol.HiddenStatesBatchRequest(prompts=["text", [1, 2]])
    assert asyncio.run(hidden_service.create_hidden_states(mixed)).code == 400
    hidden_service.supports_hidden_states = False
    request = protocol.HiddenStatesRequest(prompt="text")
    assert asyncio.run(hidden_service.create_hidden_states(request)).code == 400
    assert not hidden_service.calls
