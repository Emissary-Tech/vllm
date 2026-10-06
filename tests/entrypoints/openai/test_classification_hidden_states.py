# SPDX-License-Identifier: Apache-2.0
"""GPU integration for raw features and adapter-owned heads."""

import concurrent.futures
import json

import pytest
import requests
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ...utils import RemoteOpenAIServer

MODEL = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def classification_server(tmp_path_factory):
    directory = tmp_path_factory.mktemp("classification-heads")
    config = AutoConfig.from_pretrained(MODEL)
    hidden_size = config.hidden_size
    q_size = config.num_attention_heads * config.head_dim
    paths = {}
    weights = {}
    for index, (name, labels) in enumerate([("head-a", 2), ("head-b", 3)]):
        generator = torch.Generator().manual_seed(100 + index)
        path = directory / name
        path.mkdir()
        weight = torch.randn(labels, hidden_size, generator=generator) * 0.01
        bias = torch.arange(labels, dtype=torch.float32) + index
        prefix = "base_model.model.model.layers.0.self_attn.q_proj"
        tensors = {
            "base_model.model.score.weight": weight,
            "base_model.model.score.bias": bias,
            f"{prefix}.lora_A.weight": torch.randn(
                2, hidden_size, generator=generator
            )
            * 0.05,
            f"{prefix}.lora_B.weight": torch.randn(
                q_size, 2, generator=generator
            )
            * 0.05,
        }
        save_file(tensors, str(path / "adapter_model.safetensors"))
        (path / "adapter_config.json").write_text(
            json.dumps(
                {
                    "peft_type": "LORA",
                    "task_type": "SEQ_CLS",
                    "r": 2,
                    "lora_alpha": 2,
                    "target_modules": ["q_proj"],
                    "modules_to_save": ["score"],
                    "bias": "none",
                    "base_model_name_or_path": MODEL,
                }
            )
        )
        paths[name] = str(path)
        weights[name] = (weight.to(torch.bfloat16), bias.to(torch.bfloat16))

    args = [
        "--task=classify",
        "--enable-lora",
        "--max-lora-rank",
        "64",
        "--tensor-parallel-size",
        "1",
        "--disable-log-requests",
        "--max-loras",
        "2",
        "--max-cpu-loras",
        "4",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "512",
        "--lora-modules",
        *[f"{name}={path}" for name, path in paths.items()],
    ]
    with RemoteOpenAIServer(
        MODEL,
        args,
        env_dict={"VLLM_USE_V1": "0", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1"},
    ) as server:
        yield server, weights, paths, hidden_size


def post(server, endpoint, payload):
    response = requests.post(
        server.url_for(endpoint), json=payload, timeout=120
    )
    response.raise_for_status()
    return response.json()


def test_mixed_classification_and_hidden_requests(classification_server):
    server, weights, _, hidden_size = classification_server
    prompt = "The service was helpful and the delivery arrived early."
    expected = {}
    for model in [MODEL, "head-a", "head-b"]:
        expected[model] = post(
            server, "v1/hidden_states", {"model": model, "prompt": prompt}
        )["hidden_states"]
        assert len(expected[model]) == hidden_size
    # This also detects accidentally ignoring the requested backbone LoRA.
    assert not torch.allclose(
        torch.tensor(expected[MODEL]),
        torch.tensor(expected["head-a"]),
        atol=1e-5,
    )

    def run(index):
        model = ["head-a", "head-b", MODEL][index % 3]
        if index % 2 == 0 or model == MODEL:
            result = post(
                server, "v1/hidden_states", {"model": model, "prompt": prompt}
            )
            torch.testing.assert_close(
                torch.tensor(result["hidden_states"]),
                torch.tensor(expected[model]),
                atol=0.03,
                rtol=0.01,
            )
        else:
            result = post(
                server, "v1/classify", {"model": model, "input": prompt}
            )
            wanted = F.linear(
                torch.tensor(expected[model], dtype=torch.bfloat16),
                *weights[model],
            ).float()
            torch.testing.assert_close(
                torch.tensor(result["data"][0]["logits"]),
                wanted,
                atol=0.05,
                rtol=0.02,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        list(executor.map(run, range(36)))
    batch = post(
        server,
        "v1/hidden_states_batch",
        {"model": "head-a", "prompts": [prompt, prompt]},
    )
    for row in batch["hidden_states"]:
        torch.testing.assert_close(
            torch.tensor(row),
            torch.tensor(expected["head-a"]),
            atol=0.03,
            rtol=0.01,
        )


def test_unload_reload_and_request_validation(classification_server):
    server, _, paths, _ = classification_server
    prompt = "Adapter reload must preserve its own result."
    payload = {"model": "head-b", "input": prompt}
    before = post(server, "v1/classify", payload)["data"][0]["logits"]
    response = requests.post(
        server.url_for("v1/unload_lora_adapter"),
        json={"lora_name": "head-b"},
        timeout=120,
    )
    response.raise_for_status()
    response = requests.post(
        server.url_for("v1/hidden_states"),
        json={"model": "head-b", "prompt": prompt},
        timeout=120,
    )
    assert response.status_code == 404
    response = requests.post(
        server.url_for("v1/load_lora_adapter"),
        json={"lora_name": "head-b", "lora_path": paths["head-b"]},
        timeout=120,
    )
    response.raise_for_status()
    after = post(server, "v1/classify", payload)["data"][0]["logits"]
    torch.testing.assert_close(
        torch.tensor(after), torch.tensor(before), atol=0.03, rtol=0.01
    )
    for endpoint, body in [
        ("v1/hidden_states", {"prompt": ""}),
        ("v1/hidden_states_batch", {"prompts": []}),
    ]:
        response = requests.post(
            server.url_for(endpoint), json=body, timeout=120
        )
        assert response.status_code in (400, 422)


def test_raw_features_match_backbone(classification_server):
    server, _, _, _ = classification_server
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokens = tokenizer.encode("A short input for hidden state parity.")
    result = post(
        server, "v1/hidden_states", {"model": MODEL, "prompt": tokens}
    )
    backbone = AutoModel.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16
    ).eval()
    with torch.inference_mode():
        expected = (
            backbone(input_ids=torch.tensor([tokens]))
            .last_hidden_state[0, -1]
            .float()
        )
    torch.testing.assert_close(
        torch.tensor(result["hidden_states"]), expected, atol=0.08, rtol=0.03
    )
