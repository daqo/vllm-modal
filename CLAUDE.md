# vllm-modal

GitHub: https://github.com/daqo/vllm-modal

Deploys Qwen3.5-9B on Modal using vLLM, serving an OpenAI-compatible API with tool calling and thinking mode.

## Rules

- When making major changes (new models, features, configuration changes, or architectural updates), update `README.md` to reflect those changes.

## Stack

- **Model**: Qwen/Qwen3.5-9B (hybrid Gated DeltaNet + attention architecture, 9B params, multimodal)
- **Serving**: vLLM nightly (from wheels.vllm.ai/nightly) — nightly required because no stable release supports Qwen3.5 yet
- **Platform**: Modal (serverless GPU)
- **GPU**: H100 (single — 9B model fits comfortably)

## Commands

```bash
modal run serve.py        # Smoke test: spins up server, health check, sample query
modal deploy serve.py     # Deploy as persistent endpoint
modal app list            # Check deployed apps
```

## Key files

- `serve.py` — Main deployment script. Defines the Modal app, container image, vLLM server config, and test entrypoint.

## API usage

The endpoint is OpenAI-compatible at `/v1/chat/completions` (always `/v1`, not `/v2`).

```bash
curl -X POST https://<workspace>--qwen3-5-9b-vllm-serve.modal.run/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3.5-9B","messages":[{"role":"user","content":"Hello"}]}'
```

Tool calling is enabled — pass a `tools` array in the request body and the model will return `tool_calls` in the response.

## Features

- **Thinking mode**: `--reasoning-parser qwen3` parses `<think>...</think>` blocks
- **Tool calling**: `--enable-auto-tool-choice --tool-call-parser qwen3_coder`
- **Volume caching**: Model weights (~18GB) and vLLM compilation artifacts are cached in Modal Volumes. Cold starts reload from cache (no re-download), only needing to load weights into VRAM.

## Configuration

- `MODEL_NAME` / `MODEL_REVISION` — Model and pinned HF commit (top of serve.py). Update revision from https://huggingface.co/Qwen/Qwen3.5-9B/commits/main
- `N_GPU` — Tensor parallelism degree (1 is sufficient for 9B)
- `--max-model-len` — Context window (currently 131072, model supports up to 262144 natively, 1M+ with YaRN)
- `--enforce-eager` — Faster cold starts; remove for better throughput on long-running deployments
- `scaledown_window` — How long the container stays warm after the last request (currently 15 min)

## Upgrading vLLM to stable

Once a stable vLLM release supports Qwen3.5, check with `pip index versions vllm`, then update the image in serve.py:
```python
.pip_install("vllm==<version>", "huggingface-hub>=0.36.0")
```
Remove `pre=True` and `extra_index_url`.

## Sampling params (Qwen recommended)

- **Thinking mode**: temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5
- **Instruct (non-thinking)**: temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5
