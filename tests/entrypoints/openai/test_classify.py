# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import PretrainedConfig

from vllm.entrypoints.openai.serving_classify import (
    _get_probabilities, _should_apply_chat_template)
from vllm.outputs import PoolingOutput
from vllm.transformers_utils.config import (
    get_cross_encoder_activation_function)


def test_classify_probabilities_multiclass():
    output = PoolingOutput(torch.tensor([1.0, 2.0, 3.0]))

    assert _get_probabilities(output, "softmax") == pytest.approx(
        torch.softmax(output.data, dim=-1).tolist())


def test_classify_probabilities_multilabel():
    output = PoolingOutput(torch.tensor([-1.0, 0.0, 1.0]))

    assert _get_probabilities(output, "sigmoid") == pytest.approx(
        torch.sigmoid(output.data).tolist())


def test_classify_probabilities_binary():
    output = PoolingOutput(torch.tensor([0.75]))

    assert _get_probabilities(output, "sigmoid") == pytest.approx(
        torch.sigmoid(output.data).tolist())


def test_classify_probabilities_regression():
    output = PoolingOutput(torch.tensor([4.25]))

    assert _get_probabilities(output, "identity") == []


def test_regression_cross_encoder_activation_keeps_logits():
    config = PretrainedConfig(num_labels=1)
    config.problem_type = "regression"

    activation = get_cross_encoder_activation_function(config)

    assert isinstance(activation, torch.nn.Identity)


def test_head_only_adapter_applies_chat_template():
    assert _should_apply_chat_template({
        "target_modules": [],
        "modules_to_save": ["score"],
    })
