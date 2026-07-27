# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.lora.ops.triton_ops import utils


def test_tuned_shrink_config_forces_split_k_one(monkeypatch):
    tuned_config = {
        "1": {
            "1": {
                "1": {
                    "128": {
                        "8": {
                            "block_m": 32,
                            "block_n": 16,
                            "block_k": 128,
                            "split_k": 64,
                            "num_warps": 4,
                            "num_ctas": 1,
                            "num_stages": 2,
                        }
                    }
                }
            }
        }
    }
    monkeypatch.setattr(utils, "load_lora_op_config", lambda *_: tuned_config)
    utils.get_lora_op_configs.cache_clear()

    config = utils.get_lora_op_configs(
        "shrink",
        max_loras=1,
        batch=1,
        hidden_size=128,
        rank=8,
        num_slices=1,
    )

    assert config["split_k"] == 1
