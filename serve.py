# Serves Qwen3.5-9B on Modal using vLLM with an OpenAI-compatible API.
#
# Usage:
#   modal run serve.py        # test run: spins up server, health check, sample query
#   modal deploy serve.py     # deploy as a persistent endpoint
#
# Once deployed, use any OpenAI-compatible client pointed at the returned URL.

import json
from typing import Any

import aiohttp
import modal

# --- Model configuration ---

MODEL_NAME = "Qwen/Qwen3.5-9B"
# Pin to a specific HF commit to avoid silent model changes on redeploy.
# To update: check https://huggingface.co/Qwen/Qwen3.5-9B/commits/main
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"

VLLM_PORT = 8000
N_GPU = 1  # 9B model fits on a single H100. Increase for tensor parallelism if needed.
MINUTES = 60

# --- Container image ---
# Qwen3.5-9B uses a hybrid Gated DeltaNet + attention architecture that isn't in
# any stable vLLM release yet. We pull from the nightly wheel index instead.
# Once a stable release supports Qwen3.5, replace with:
#   .pip_install("vllm==<version>", "huggingface-hub>=0.36.0")
# and remove pre=True and extra_index_url.

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .pip_install(
        "vllm",
        "huggingface-hub>=0.36.0",
        pre=True,  # needed to install nightly pre-release builds
        extra_index_url="https://wheels.vllm.ai/nightly",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})  # faster model weight downloads
)

# --- Persistent storage ---
# Cache model weights and vLLM compilation artifacts across container restarts
# so cold starts don't re-download ~18GB every time.

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("qwen3.5-9b-vllm")

# --- vLLM server ---
# Spawns a vLLM process that serves an OpenAI-compatible API on port 8000.
# Modal proxies external HTTPS traffic to this port automatically.
#
# On cold start: container boots, vLLM loads weights from cached volumes into GPU
# memory, then starts accepting requests. After 15 min with no requests, the
# container shuts down — but volumes persist, so the next cold start skips the
# download and only needs to reload into VRAM.

@app.function(
    image=vllm_image,
    gpu=f"H100:{N_GPU}",
    scaledown_window=15 * MINUTES,  # keep container warm for 15 min after last request
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=32)  # vLLM handles batching internally
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--uvicorn-log-level=info",
        "--tensor-parallel-size",
        str(N_GPU),
        # Model supports up to 262144 natively (1M+ with YaRN).
        "--max-model-len",
        "131072",
        # Enables parsing of <think>...</think> blocks in model output.
        "--reasoning-parser",
        "qwen3",
        # Skip Torch compilation and CUDA graph capture for faster cold starts.
        # Remove this flag for better throughput on long-running deployments.
        "--enforce-eager",
    ]

    print("Starting vLLM:", " ".join(cmd))
    subprocess.Popen(" ".join(cmd), shell=True)


# --- Test entrypoint ---
# Runs locally, sends requests to the remote Modal server.
# Useful for quick smoke tests: `modal run serve.py`

@app.local_entrypoint()
async def test(test_timeout=10 * MINUTES, content=None):
    url = await serve.get_web_url.aio()

    if content is None:
        content = "What is the capital of France? Keep it brief."

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content},
    ]

    async with aiohttp.ClientSession(base_url=url) as session:
        # Wait for the server to be ready (model loading can take a few minutes)
        print(f"Health check: {url}")
        async with session.get(
            "/health", timeout=aiohttp.ClientTimeout(total=test_timeout - MINUTES)
        ) as resp:
            assert resp.status == 200, f"Health check failed: {resp.status}"
            print("Health check passed")

        # Send a streaming chat completion request using Qwen's recommended sampling params
        print(f"Sending: {content}")
        payload: dict[str, Any] = {
            "messages": messages,
            "model": MODEL_NAME,
            "stream": True,
            "max_tokens": 4096,
            "temperature": 1.0,
            "top_p": 0.95,
            "presence_penalty": 1.5,
            "top_k": 20,
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        # Stream and print tokens as they arrive (SSE protocol)
        async with session.post(
            "/v1/chat/completions", json=payload, headers=headers
        ) as resp:
            resp.raise_for_status()
            async for raw in resp.content:
                line = raw.decode().strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    line = line[len("data: "):]
                chunk = json.loads(line)
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    print(delta, end="", flush=True)
        print()
