# SPDX-License-Identifier: Apache-2.0
"""
Debug utilities for comparing layer outputs across multiple runs.
Enable with: VLLM_DEBUG_LAYERS=1
"""

import os
import torch
import json
import tempfile
from pathlib import Path
from typing import Dict, Optional
from collections import defaultdict

# Global debug state
VLLM_DEBUG = os.environ.get("VLLM_DEBUG_LAYERS", "0") == "1"
VLLM_DEBUG_DIR = Path(tempfile.gettempdir()) / "vllm_debug"
VLLM_DEBUG_MAX_RUNS = 5

# Create debug directory if enabled
if VLLM_DEBUG:
    VLLM_DEBUG_DIR.mkdir(exist_ok=True)


def _get_run_id_file() -> Path:
    return VLLM_DEBUG_DIR / "run_id.txt"


def _get_tensor_file(run_id: int, name: str) -> Path:
    safe_name = name.replace("/", "_")
    return VLLM_DEBUG_DIR / f"run_{run_id}_{safe_name}.pt"


def is_debug_enabled() -> bool:
    return VLLM_DEBUG


def save_debug_output(name: str, tensor: torch.Tensor) -> None:
    """Save tensor output for debugging to file."""
    if not VLLM_DEBUG:
        return

    run_id = get_current_run_id()
    tensor_file = _get_tensor_file(run_id, name)
    # Clone and convert to float32 for consistent comparison
    torch.save(tensor.detach().float().cpu().clone(), tensor_file)


def increment_run_id() -> int:
    """Increment run ID after each forward pass."""
    current = get_current_run_id()
    new_id = (current + 1) % VLLM_DEBUG_MAX_RUNS
    _get_run_id_file().write_text(str(new_id))
    return new_id


def get_current_run_id() -> int:
    """Get current run ID."""
    run_id_file = _get_run_id_file()
    if run_id_file.exists():
        try:
            return int(run_id_file.read_text().strip())
        except:
            return 0
    return 0


def reset_debug_state() -> None:
    """Reset all debug state."""
    if VLLM_DEBUG_DIR.exists():
        for f in VLLM_DEBUG_DIR.glob("*.pt"):
            f.unlink()
        run_id_file = _get_run_id_file()
        if run_id_file.exists():
            run_id_file.unlink()


def get_debug_outputs() -> Dict[str, torch.Tensor]:
    """Get all captured outputs from files."""
    outputs = {}
    if not VLLM_DEBUG_DIR.exists():
        return outputs

    for f in VLLM_DEBUG_DIR.glob("run_*.pt"):
        # Extract key from filename: run_0_layer_1.pt -> run_0_layer_1
        key = f.stem
        try:
            outputs[key] = torch.load(f, weights_only=True)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    return outputs


def compare_runs(run_id_1: int = 0, run_id_2: int = 1, threshold: float = 1e-6) -> Dict:
    """
    Compare outputs between two runs.

    Returns dict with comparison results for each layer.
    """
    # Load all debug outputs from files
    debug_outputs = get_debug_outputs()

    results = {}

    # Get all layer names from run 0
    prefix_1 = f"run_{run_id_1}_"
    prefix_2 = f"run_{run_id_2}_"

    layer_names = set()
    for key in debug_outputs.keys():
        if key.startswith(prefix_1):
            layer_names.add(key.replace(prefix_1, ""))

    for layer_name in sorted(layer_names):
        key_1 = f"{prefix_1}{layer_name}"
        key_2 = f"{prefix_2}{layer_name}"

        if key_1 not in debug_outputs or key_2 not in debug_outputs:
            results[layer_name] = {"status": "missing", "details": "One of the runs missing this layer"}
            continue

        tensor_1 = debug_outputs[key_1]
        tensor_2 = debug_outputs[key_2]

        if tensor_1.shape != tensor_2.shape:
            results[layer_name] = {
                "status": "shape_mismatch",
                "shape_1": list(tensor_1.shape),
                "shape_2": list(tensor_2.shape)
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
    # Load all debug outputs from files
    debug_outputs = get_debug_outputs()

    print("\n" + "="*80)
    print("Comparing all runs against Run 0")
    print("="*80)

    # Find all run IDs
    run_ids = set()
    for key in debug_outputs.keys():
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
