from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.v1.pool.metadata import PoolingMetadata

logger = init_logger(__name__)


@dataclass
class AdapterClassificationHead:
    module_name: str
    weight: torch.Tensor
    bias: torch.Tensor | None = None

    def clone(self) -> "AdapterClassificationHead":
        return AdapterClassificationHead(
            module_name=self.module_name,
            weight=self.weight,
            bias=self.bias,
        )


class DynamicClassificationHead(nn.Module):
    """Dispatch classification heads per request using LoRA metadata."""

    def __init__(self, base_head: nn.Module):
        super().__init__()
        self.base_head = base_head
        self._active_heads: dict[int, AdapterClassificationHead] = {}

    def clear_adapter_heads(self) -> None:
        self._active_heads.clear()

    def remove_adapter_head(self, adapter_id: int) -> None:
        self._active_heads.pop(adapter_id, None)

    def set_adapter_head(
        self,
        adapter_id: int,
        head: AdapterClassificationHead,
    ) -> None:
        weight = self._get_weight()
        bias = self._get_bias()

        if head.weight.ndim != 2 or weight.ndim != 2:
            raise ValueError(
                "Classification head shape mismatch: "
                f"adapter has {tuple(head.weight.shape)}, "
                f"but model has {tuple(weight.shape)}."
            )
        if head.weight.shape[1] != weight.shape[1]:
            raise ValueError(
                "Classification head hidden dimension mismatch: "
                f"adapter has {tuple(head.weight.shape)}, "
                f"but model has {tuple(weight.shape)}."
            )

        adapter_bias = head.bias
        if adapter_bias is not None and adapter_bias.ndim != 1:
            raise ValueError(
                "Classification bias shape mismatch: "
                f"adapter has {tuple(adapter_bias.shape)}."
            )
        if adapter_bias is not None and adapter_bias.shape[0] != head.weight.shape[0]:
            raise ValueError(
                "Classification bias/output dimension mismatch: "
                f"adapter bias has {tuple(adapter_bias.shape)}, "
                f"but adapter weight has {tuple(head.weight.shape)}."
            )

        self._active_heads[adapter_id] = AdapterClassificationHead(
            module_name=head.module_name,
            weight=head.weight.to(device=weight.device, dtype=weight.dtype),
            bias=None
            if adapter_bias is None
            else adapter_bias.to(
                device=weight.device,
                dtype=weight.dtype if bias is None else bias.dtype,
            ),
        )

    def forward(self, pooled_data: torch.Tensor) -> torch.Tensor:
        output = self.base_head(pooled_data)
        return output[0] if isinstance(output, tuple) else output

    def forward_with_pooling_metadata(
        self,
        pooled_data: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> torch.Tensor | list[torch.Tensor]:
        lora_ids = [
            int((params.extra_kwargs or {}).get("lora_int_id", 0))
            for params in pooling_metadata.pooling_params
        ]
        if not lora_ids or all(lora_id == lora_ids[0] for lora_id in lora_ids):
            return self._apply_head(pooled_data, lora_ids[0] if lora_ids else 0)

        per_request_outputs: list[torch.Tensor | None] = [None] * pooled_data.shape[0]
        unique_lora_ids = sorted(set(lora_ids))
        for lora_id in unique_lora_ids:
            indices = [
                idx for idx, current_id in enumerate(lora_ids) if current_id == lora_id
            ]
            index_tensor = torch.tensor(
                indices, device=pooled_data.device, dtype=torch.long
            )
            grouped_output = self._apply_head(
                pooled_data.index_select(0, index_tensor), lora_id
            )
            for out_index, row in zip(indices, grouped_output):
                per_request_outputs[out_index] = row

        outputs = [out for out in per_request_outputs if out is not None]
        if not outputs:
            return []

        output_shapes = {tuple(out.shape) for out in outputs}
        if len(output_shapes) == 1:
            return torch.stack(outputs)
        return outputs

    def _apply_head(self, pooled_data: torch.Tensor, adapter_id: int) -> torch.Tensor:
        adapter_head = self._active_heads.get(adapter_id)
        if adapter_head is None:
            return self.forward(pooled_data)

        return F.linear(pooled_data, adapter_head.weight, adapter_head.bias)

    def _get_bias(self) -> torch.Tensor | None:
        bias = getattr(self.base_head, "bias", None)
        return bias if isinstance(bias, torch.Tensor) else None

    def _get_weight(self) -> torch.Tensor:
        weight = getattr(self.base_head, "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise ValueError(
                "Classification head "
                f"{type(self.base_head).__name__} "
                "does not expose a weight tensor."
            )
        return weight
