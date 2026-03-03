# vllm-modal

Deploy [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) on [Modal](https://modal.com) using [vLLM](https://docs.vllm.ai), serving an OpenAI-compatible API.

## What this does

Runs Qwen3.5-9B on a single H100 GPU via Modal's serverless infrastructure. The server exposes an OpenAI-compatible `/v1/chat/completions` endpoint with:

- **Thinking mode** — model outputs reasoning in `<think>` blocks before responding
- **Tool calling** — pass a `tools` array and the model returns structured `tool_calls`
- **131k context** — supports up to 131,072 tokens (model natively supports 262k)

## Supported model

| Model | Parameters | Architecture | Context | License |
|-------|-----------|--------------|---------|---------|
| [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 9B | Hybrid Gated DeltaNet + Attention | 262k native | Apache 2.0 |

Qwen3.5-9B is a natively multimodal model (text, image, video) with multi-token prediction, supporting 201 languages.

## Quick start

```bash
pip install modal
modal setup

# Test run (spins up server, runs health check and sample query)
modal run serve.py

# Deploy as a persistent endpoint
modal deploy serve.py
```

## Usage

Once deployed, send requests with any OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<your-workspace>--qwen3-5-9b-vllm-serve.modal.run/v1",
    api_key="EMPTY",
)

response = client.chat.completions.create(
    model="Qwen/Qwen3.5-9B",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Tool calling

```python
response = client.chat.completions.create(
    model="Qwen/Qwen3.5-9B",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }],
)
```

## Configuration

Key settings in `serve.py`:

| Setting | Current value | Description |
|---------|--------------|-------------|
| `N_GPU` | 1 | Tensor parallelism degree |
| `--max-model-len` | 131072 | Context window size |
| `--enforce-eager` | enabled | Faster cold starts, lower throughput |
| `scaledown_window` | 15 min | How long container stays warm after last request |
