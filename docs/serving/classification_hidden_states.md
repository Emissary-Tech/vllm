# Classification heads and hidden states

This branch adds raw last-token features to the converted classification model
used by `Qwen/Qwen3-8B`. It also keeps each adapter's classification or regression
head separate. A mixed batch selects the head using the scheduler's LoRA ID for
each request. Loading an adapter no longer replaces the shared `model.score`.

## Install and start

Install a build from `feature/add-classification-head-hs` in the existing Linux
GPU environment. The published `0.8.4+emissary.zeroshot.precompiled` wheel does not
contain these changes. From a checkout of this branch, the existing precompiled
build workflow can be used:

```bash
VLLM_USE_PRECOMPILED=1 uv pip install .
```

The existing launch command works:

```bash
vllm serve "$BASE_MODEL" \
  --task=classify \
  --enable-lora \
  --max-lora-rank 64 \
  --tensor-parallel-size 1 \
  --disable-log-requests
```

For several adapters to execute in the same GPU batch, configure enough slots:

```bash
vllm serve "$BASE_MODEL" \
  --task=classify \
  --enable-lora \
  --max-lora-rank 64 \
  --max-loras 8 \
  --max-cpu-loras 32 \
  --tensor-parallel-size 1 \
  --disable-log-requests \
  --lora-modules classifier-a=/path/to/adapter-a classifier-b=/path/to/adapter-b
```

`max_loras` controls simultaneously active GPU adapters. Merely registering
several adapters does not increase that limit. Head GPU copies follow the same
activation/eviction lifecycle; cached CPU adapter weights are reused on reload.
Choose the slot count for the available GPU memory.

The hidden-state API uses the normal pooling scheduler and engine transport.
It does not require `--return-hidden-states`, one-token generation, or
`--disable-frontend-multiprocessing`.

## Hidden-state API

`POST /v1/hidden_states` returns one vector:

```bash
curl -sS http://localhost:8000/v1/hidden_states \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Already rendered prompt"}'
```

```json
{"hidden_states": [0.1, -0.2, 0.3]}
```

`POST /v1/hidden_states_batch` returns vectors in input order:

```bash
curl -sS http://localhost:8000/v1/hidden_states_batch \
  -H 'Content-Type: application/json' \
  -d '{"model":"classifier-a","prompts":["Rendered prompt one","Rendered prompt two"]}'
```

```json
{"hidden_states": [[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]]}
```

The displayed vectors are abbreviated examples. The actual length is the
backbone hidden size. `model` can be the base model or a registered LoRA name;
omitting it selects the base model. Selecting a LoRA applies its backbone LoRA
weights while bypassing its score head, regression centering, cosine
normalization, and output activation. Unregistered model names return 404.
Adapter paths/IDs are not accepted in these requests: register an adapter with
the existing LoRA loading API or `--lora-modules`, then use its name in `model`.

Supported options are `model`, `add_special_tokens` (default true),
`truncate_prompt_tokens`, and `priority`. A prompt can also be a nonempty list of
nonnegative token IDs. Batches must contain all strings or all token-ID lists.
Empty batches and invalid prompts are rejected.

The caller must supply the same fully rendered prompt used to build/train the
head. The endpoint does **not** apply a chat template or the adapter's class
description prompt. For Playground, preserve its current class/regression prompt
and tokenizer chat template, including `enable_thinking=False` and
`add_generation_prompt=True`. Exact token IDs can be supplied to avoid changes in
tokenization or duplicated special tokens.

The returned features are the backbone's final output at the last input token,
including the backbone's own final normalization, but before any **head-specific**
processing. The headless path does not need `lm_head`. Native pooling/classifier
architectures that have not been converted through `as_classification_model`
are rejected by this endpoint.

## CPU init and quick-train

1. For regression init, render the baseline, low endpoint, and high endpoint
   prompts on the CPU server. Submit them with `/v1/hidden_states_batch`.
2. Calculate the head on the CPU server using the returned vectors. Preserve
   `score.weight`, `score.bias`, `regression.v_base`, and the existing regression
   metadata in the exported adapter.
3. For classification/regression quick-train, obtain the training samples'
   features from the same endpoint, using the same backbone/LoRA and prompt
   format that will be used for inference.
4. Register the exported adapter and use the existing `/v1/classify` endpoint
   for inference (`classification_type="regression"` for regression).

Cache features or heads only under the matching base-model revision, tokenizer
and prompt configuration, backbone LoRA revision, and complete task definition.

Existing head-only multiclass adapters retain cosine scoring with temperature
40. Binary heads and ordinary LoRA classifiers retain linear scoring. Regression
retains baseline centering, L2 normalization, weight/bias projection, and existing
response clipping. Base-model requests use the base head rather than the last
adapter that happened to be loaded.

## Validation

CPU math, routing, request validation, and serialized pooling flags:

```bash
python -m pytest --confcutdir=tests/classification_cpu tests/classification_cpu -q
```

In a compatible Linux vLLM environment, test the adapter lifecycle and run the
GPU integration suite (downloads `Qwen/Qwen3-0.6B`):

```bash
VLLM_USE_V1=0 python -m pytest \
  tests/lora/test_classification_head_lifecycle.py \
  tests/entrypoints/openai/test_classification_hidden_states.py -v
```

The GPU suite covers real backbone LoRAs with different head dimensions,
concurrent classification and raw-feature requests, batch extraction,
unload/reload, invalid requests, and comparison with the Hugging Face backbone.
Run production-size Qwen3-8B parity and workload tests before deploying a new
wheel; these changes have not been benchmarked on that model.
