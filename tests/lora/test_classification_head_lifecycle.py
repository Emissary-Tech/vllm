# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm.lora.models import LoRAModel, LoRAModelManager
from vllm.model_executor.layers.classification_head import ClassificationHead


def make_manager():
    # Exercise the real cache/activation methods without allocating Punica or
    # loading a backbone. Projection kernels are covered by GPU serving tests.
    manager = object.__new__(LoRAModelManager)
    manager.model = SimpleNamespace(
        score=torch.nn.Linear(3, 2, bias=False),
        supports_classification_hidden_states=True,
    )
    manager.device = torch.device("cpu")
    manager.lora_config = SimpleNamespace(
        max_loras=2, max_cpu_loras=3, bias_enabled=False
    )
    manager._active_adapters = {}
    manager._registered_adapters = {}
    manager._classification_heads = {}
    manager.lora_index_to_id = [None, None]
    manager.modules = {}
    return manager


def test_activate_switch_evict_reload_and_remove_heads():
    manager = make_manager()
    original = manager.model.score.weight.detach().clone()
    a = LoRAModel(
        1, 1, {}, classification_head=ClassificationHead(torch.ones(2, 3))
    )
    b = LoRAModel(
        2, 1, {}, classification_head=ClassificationHead(-torch.ones(1, 3))
    )
    manager._registered_adapters.update({1: a, 2: b})
    assert manager.activate_adapter(1)
    assert manager.activate_adapter(2)
    assert not manager.activate_adapter(1)
    assert manager.get_classification_head(1).weight.shape == (2, 3)
    assert manager.get_classification_head(2).weight.shape == (1, 3)
    torch.testing.assert_close(manager.model.score.weight, original)

    manager.deactivate_adapter(1)
    assert 1 not in manager._classification_heads
    with pytest.raises(ValueError, match="not active"):
        manager.get_classification_head(1)
    assert manager.activate_adapter(1)
    torch.testing.assert_close(
        manager.get_classification_head(1).weight, a.classification_head.weight
    )

    assert manager.remove_adapter(2)
    replacement = LoRAModel(
        2,
        1,
        {},
        classification_head=ClassificationHead(torch.full((4, 3), 7.0)),
    )
    manager._registered_adapters[2] = replacement
    assert manager.activate_adapter(2)
    assert manager.get_classification_head(2).weight.shape == (4, 3)
    manager.remove_all_adapters()
    assert not manager._classification_heads
    assert not manager._active_adapters
    torch.testing.assert_close(manager.model.score.weight, original)


def test_clone_preserves_adapter_head_and_plain_lora_has_no_head():
    head = ClassificationHead(torch.ones(1, 3), torch.zeros(1), torch.ones(3))
    a = LoRAModel(1, 1, {}, scaling_factor=2.0, classification_head=head)
    clone = a.clone(2)
    assert clone.id == 2
    assert clone.classification_head is head
    assert clone.scaling_factor == 2.0
    manager = make_manager()
    manager._registered_adapters[3] = LoRAModel(3, 1, {})
    manager.activate_adapter(3)
    assert manager.get_classification_head(3) is None


def test_invalid_head_cannot_mark_adapter_active():
    manager = make_manager()
    manager._registered_adapters[1] = LoRAModel(
        1, 1, {}, classification_head=ClassificationHead(torch.ones(2, 4))
    )
    with pytest.raises(ValueError, match="hidden size"):
        manager.activate_adapter(1)
    assert not manager._active_adapters
