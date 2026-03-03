# vllm-modal

Deploys Qwen3.5-9B on Modal using vLLM, serving an OpenAI-compatible API.

## Stack

- **Model**: Qwen/Qwen3.5-9B (hybrid Gated DeltaNet + attention architecture)
- **Serving**: vLLM nightly (from wheels.vllm.ai/nightly)
- **Platform**: Modal (serverless GPU)
- **GPU**: H100 (single)

## Commands

```bash
modal run serve.py        # Test: spins up server, health check, sample query
modal deploy serve.py     # Deploy as persistent endpoint
```

## Key files

- `serve.py` — Main deployment script. Defines the Modal app, container image, vLLM server config, and test entrypoint.

## Configuration

- `MODEL_NAME` / `MODEL_REVISION` — Model to serve (top of serve.py)
- `N_GPU` — Tensor parallelism degree
- `--max-model-len` — Context window (currently 32768, model supports up to 262144)
- `--enforce-eager` — Faster cold starts; remove for better throughput
- `--reasoning-parser qwen3` — Enables thinking mode support
