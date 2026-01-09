# SPDX-License-Identifier: Apache-2.0
"""
Debug utilities for comparing layer outputs across multiple runs.
Enable with: VLLM_DEBUG_LAYERS=1
"""

import os
import torch
from typing import Dict, Optional
from collections import defaultdict

# Global debug state
VLLM_DEBUG = os.environ.get("VLLM_DEBUG_LAYERS", "0") == "1"
VLLM_DEBUG_OUTPUTS: Dict[str, torch.Tensor] = {}
VLLM_DEBUG_RUN_ID = 0
VLLM_DEBUG_MAX_RUNS = 5


def is_debug_enabled() -> bool:
    return VLLM_DEBUG


def save_debug_output(name: str, tensor: torch.Tensor) -> None:
    """Save tensor output for debugging."""
    if not VLLM_DEBUG:
        return

    global VLLM_DEBUG_RUN_ID
    key = f"run_{VLLM_DEBUG_RUN_ID}_{name}"
    # Clone and convert to float32 for consistent comparison
    VLLM_DEBUG_OUTPUTS[key] = tensor.detach().float().cpu().clone()


def increment_run_id() -> int:
    """Increment run ID after each forward pass."""
    global VLLM_DEBUG_RUN_ID
    VLLM_DEBUG_RUN_ID = (VLLM_DEBUG_RUN_ID + 1) % VLLM_DEBUG_MAX_RUNS
    return VLLM_DEBUG_RUN_ID


def reset_debug_state() -> None:
    """Reset all debug state."""
    global VLLM_DEBUG_RUN_ID, VLLM_DEBUG_OUTPUTS
    VLLM_DEBUG_RUN_ID = 0
    VLLM_DEBUG_OUTPUTS = {}


def get_debug_outputs() -> Dict[str, torch.Tensor]:
    """Get all captured outputs."""
    return VLLM_DEBUG_OUTPUTS


def get_current_run_id() -> int:
    """Get current run ID."""
    return VLLM_DEBUG_RUN_ID


def compare_runs(run_id_1: int = 0, run_id_2: int = 1, threshold: float = 1e-6) -> Dict:
    """
    Compare outputs between two runs.

    Returns dict with comparison results for each layer.
    """
    results = {}

    # Get all layer names from run 0
    prefix_1 = f"run_{run_id_1}_"
    prefix_2 = f"run_{run_id_2}_"

    layer_names = set()
    for key in VLLM_DEBUG_OUTPUTS.keys():
        if key.startswith(prefix_1):
            layer_names.add(key.replace(prefix_1, ""))

    for layer_name in sorted(layer_names):
        key_1 = f"{prefix_1}{layer_name}"
        key_2 = f"{prefix_2}{layer_name}"

        if key_1 not in VLLM_DEBUG_OUTPUTS or key_2 not in VLLM_DEBUG_OUTPUTS:
            results[layer_name] = {"status": "missing", "details": "One of the runs missing this layer"}
            continue

        tensor_1 = VLLM_DEBUG_OUTPUTS[key_1]
        tensor_2 = VLLM_DEBUG_OUTPUTS[key_2]

        if tensor_1.shape != tensor_2.shape:
            results[layer_name] = {
                "status": "shape_mismatch",
                "shape_1": tensor_1.shape,
                "shape_2": tensor_2.shape
            }
            continue

        diff = (tensor_1 - tensor_2).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        std_diff = diff.std().item()

        # Find where the max diff occurs
        max_idx = diff.argmax().item()

        is_different = max_diff > threshold

        results[layer_name] = {
            "status": "different" if is_different else "same",
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "max_diff_idx": max_idx,
            "value_1_at_max": tensor_1.flatten()[max_idx].item(),
            "value_2_at_max": tensor_2.flatten()[max_idx].item(),
            "shape": list(tensor_1.shape),
        }

    return results


def print_comparison_report(run_id_1: int = 0, run_id_2: int = 1, threshold: float = 1e-6) -> None:
    """Print a human-readable comparison report."""
    results = compare_runs(run_id_1, run_id_2, threshold)

    print(f"\n{'='*80}")
    print(f"VLLM Layer Comparison Report: Run {run_id_1} vs Run {run_id_2}")
    print(f"Threshold: {threshold}")
    print(f"{'='*80}\n")

    first_divergence = None

    for layer_name, data in results.items():
        status = data["status"]

        if status == "same":
            print(f"[OK] {layer_name}: identical (max_diff={data['max_diff']:.2e})")
        elif status == "different":
            if first_divergence is None:
                first_divergence = layer_name
            print(f"[DIFF] {layer_name}:")
            print(f"       max_diff={data['max_diff']:.6e}, mean_diff={data['mean_diff']:.6e}")
            print(f"       at index {data['max_diff_idx']}: {data['value_1_at_max']:.6f} vs {data['value_2_at_max']:.6f}")
        elif status == "missing":
            print(f"[MISSING] {layer_name}: {data['details']}")
        elif status == "shape_mismatch":
            print(f"[SHAPE] {layer_name}: {data['shape_1']} vs {data['shape_2']}")

    print(f"\n{'='*80}")
    if first_divergence:
        print(f"First divergence detected at: {first_divergence}")
    else:
        print("No divergence detected - all layers identical!")
    print(f"{'='*80}\n")


def compare_all_runs(threshold: float = 1e-6) -> None:
    """Compare all captured runs against run 0."""
    print("\n" + "="*80)
    print("Comparing all runs against Run 0")
    print("="*80)

    # Find all run IDs
    run_ids = set()
    for key in VLLM_DEBUG_OUTPUTS.keys():
        parts = key.split("_")
        if len(parts) >= 2 and parts[0] == "run":
            try:
                run_ids.add(int(parts[1]))
            except ValueError:
                pass

    run_ids = sorted(run_ids)
    if len(run_ids) < 2:
        print("Need at least 2 runs to compare")
        return

    base_run = run_ids[0]
    for other_run in run_ids[1:]:
        print(f"\n--- Run {base_run} vs Run {other_run} ---")
        results = compare_runs(base_run, other_run, threshold)

        different_layers = [k for k, v in results.items() if v.get("status") == "different"]
        if different_layers:
            print(f"Different layers: {different_layers[:5]}...")  # Show first 5
            first_diff = results[different_layers[0]]
            print(f"First diff ({different_layers[0]}): max={first_diff['max_diff']:.6e}")
        else:
            print("All layers identical!")
