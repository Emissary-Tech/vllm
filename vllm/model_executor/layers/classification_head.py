# SPDX-License-Identifier: Apache-2.0
"""Adapter-owned classification weights, independent of the shared model."""

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F

REGRESSION_V_BASE_KEY = "base_model.model.regression.v_base"


@dataclass(frozen=True)
class ClassificationHead:
    weight: torch.Tensor
    bias: Optional[torch.Tensor] = None
    v_base: Optional[torch.Tensor] = None
    use_cosine: bool = False
    temperature: float = 40.0

    @classmethod
    def from_tensors(
        cls, tensors: dict[str, torch.Tensor], *, head_only: bool
    ) -> Optional["ClassificationHead"]:
        weights = [
            value
            for name, value in tensors.items()
            if name.endswith(("score.weight", "classifier.weight"))
        ]
        biases = [
            value
            for name, value in tensors.items()
            if name.endswith(("score.bias", "classifier.bias"))
        ]
        v_base = tensors.get(REGRESSION_V_BASE_KEY)
        if not weights:
            if biases or v_base is not None:
                raise ValueError(
                    "Classification bias/v_base requires a head weight"
                )
            return None
        if len(weights) != 1 or len(biases) > 1:
            raise ValueError("Expected one classification head per adapter")
        weight = weights[0]
        bias = biases[0] if biases else None
        if weight.ndim != 2 or min(weight.shape) < 1:
            raise ValueError(
                "Classification weight must have shape [labels, hidden_size]"
            )
        if bias is not None and bias.shape != (weight.shape[0],):
            raise ValueError(
                "Classification bias does not match the head output size"
            )
        if v_base is not None and (
            v_base.shape != (weight.shape[1],) or weight.shape[0] != 1
        ):
            raise ValueError(
                "Regression requires one output and a hidden_size v_base"
            )
        return cls(
            weight.detach(),
            None if bias is None else bias.detach(),
            None if v_base is None else v_base.detach(),
            head_only and v_base is None and weight.shape[0] > 1,
        )

    def to(
        self, device: torch.device, dtype: torch.dtype
    ) -> "ClassificationHead":
        return ClassificationHead(
            self.weight.to(device=device, dtype=dtype),
            None
            if self.bias is None
            else self.bias.to(device=device, dtype=dtype),
            # Centering is performed in float32, just as in Playground.
            None
            if self.v_base is None
            else self.v_base.to(device=device, dtype=torch.float32),
            self.use_cosine,
            self.temperature,
        )

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[-1] != self.weight.shape[1]:
            raise ValueError(
                "Hidden state and classification head dimensions differ"
            )
        if self.v_base is not None:
            features = F.normalize(hidden_states.float() - self.v_base, dim=-1)
            return F.linear(
                features.to(self.weight.dtype), self.weight, self.bias
            )
        if self.use_cosine:
            # Preserve the existing head-only multiclass scoring contract.
            return self.temperature * F.linear(
                F.normalize(hidden_states.float(), dim=-1),
                F.normalize(self.weight.float(), dim=-1),
            )
        return F.linear(
            hidden_states.to(self.weight.dtype), self.weight, self.bias
        )


def classify_rows(
    hidden_states: torch.Tensor,
    lora_ids: list[int],
    return_hidden_states: list[bool],
    get_head: Callable[[int], Optional[ClassificationHead]],
    base_score: Callable[[torch.Tensor], torch.Tensor],
) -> list[torch.Tensor]:
    """Route pooled rows by adapter, allowing different output dimensions.

    Raw features and base-model requests never consult another adapter's head.
    Requests for the same head share one projection, regardless of batch order.
    """
    if len(lora_ids) != len(hidden_states) or len(return_hidden_states) != len(
        lora_ids
    ):
        raise ValueError(
            "Classification metadata does not match the pooled batch"
        )
    outputs: list[Optional[torch.Tensor]] = [None] * len(lora_ids)
    groups: dict[int, list[int]] = {}
    for index, (lora_id, raw) in enumerate(zip(lora_ids, return_hidden_states)):
        if raw:
            outputs[index] = hidden_states[index].float()
        else:
            groups.setdefault(lora_id, []).append(index)
    for lora_id, indices in groups.items():
        head = get_head(lora_id) if lora_id else None
        features = hidden_states[indices]
        logits = head(features) if head is not None else base_score(features)
        for index, row in zip(indices, logits):
            outputs[index] = row
    assert all(output is not None for output in outputs)
    return [output for output in outputs if output is not None]
