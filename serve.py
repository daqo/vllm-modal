import json
from typing import Any

import aiohttp
import modal

MODEL_NAME = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "main"
VLLM_PORT = 8000
N_GPU = 1
MINUTES = 60

# Qwen3.5-9B requires nightly vLLM for architecture support (Gated DeltaNet + hybrid attention)
vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .pip_install(
        "vllm",
        "huggingface-hub>=0.36.0",
        pre=True,
        extra_index_url="https://wheels.vllm.ai/nightly",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("qwen3.5-9b-vllm")


@app.function(
    image=vllm_image,
    gpu=f"H100:{N_GPU}",
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=32)
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
        "--max-model-len",
        "32768",
        "--reasoning-parser",
        "qwen3",
        "--enforce-eager",
    ]

    print("Starting vLLM:", " ".join(cmd))
    subprocess.Popen(" ".join(cmd), shell=True)


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
        print(f"Health check: {url}")
        async with session.get(
            "/health", timeout=aiohttp.ClientTimeout(total=test_timeout - MINUTES)
        ) as resp:
            assert resp.status == 200, f"Health check failed: {resp.status}"
            print("Health check passed")

        print(f"Sending: {content}")
        payload: dict[str, Any] = {
            "messages": messages,
            "model": MODEL_NAME,
            "stream": True,
            "max_tokens": 4096,
            "temperature": 1.0,
            "top_p": 0.95,
            "presence_penalty": 1.5,
            "extra_body": {"top_k": 20},
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

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
