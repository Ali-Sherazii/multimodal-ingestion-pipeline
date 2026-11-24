"""The pipeline's own language models on a Modal GPU, served by vLLM: Gemma 4 26B (app
"ingest-llm": chat, router, memory) and Qwen 3.8 27B (app "ingest-llm-qwen": agents).

Google's free Gemini key allows ~20 good answers a day; these copies have no daily limit and
nothing sent to them leaves your Modal account. They cost GPU time instead (an A100 80 GB is
about $2.50 an hour, billed per second, only while it runs).

Set up each once (the weights go into the Modal volume "ingest-llm-weights" on a cheap CPU
machine, so no GPU time is spent downloading); INGEST_LLM_SERVER picks the model:

    modal run src/ingest/llm_modal.py::download            # Gemma (52 GB)
    modal deploy src/ingest/llm_modal.py
    INGEST_LLM_SERVER=qwen modal run src/ingest/llm_modal.py::download   # Qwen (31 GB)
    INGEST_LLM_SERVER=qwen modal deploy src/ingest/llm_modal.py

A deployed app costs nothing while idle. The first request after a quiet spell starts a
GPU and loads the model (a few minutes); the GPU stops SCALEDOWN_SECONDS after the last
request. There is no public web address: requests go through Modal's own API with your
Modal login, so nobody else can start the GPU or spend your credits.

Requests and replies use the OpenAI chat-completions format (messages, tools,
response_format, max_tokens, temperature, chat_template_kwargs) and are passed to vLLM
unchanged. This file is self-contained: the container needs only vLLM's image.
"""
import json
import os
import subprocess
import time
from dataclasses import dataclass

import modal


@dataclass(frozen=True)
class Spec:
    app: str            # the deployed Modal app
    served: str         # the model name the pipeline asks for (and shows people)
    model: str          # Hugging Face repository
    revision: str       # pinned: the same weights every run
    gpu: str
    context: int        # tokens per request (prompt + reply)
    args: tuple = ()    # vLLM options for this model (tool calls, thinking, text only)


MODELS = {
    # 26B mixture of experts (~4B active per word): fast (~120 tokens/s); 52 GB
    "gemma": Spec("ingest-llm", "gemma-4-26b", "google/gemma-4-26B-A4B-it",
                  "4d7ae4984b7db7de8f8457170b3f1a419ee76d52", "A100-80GB", 65536,
                  ("--tool-call-parser", "gemma4", "--reasoning-parser", "gemma4",
                   "--limit-mm-per-prompt", json.dumps({"image": 0, "audio": 0, "video": 0}))),
    # 27B dense, built for long multi-step agent work; Qwen's own FP8 version (31 GB) leaves
    # room for long agent conversations (most of its layers keep no per-word memory)
    "qwen": Spec("ingest-llm-qwen", "qwen3.8-27b", "Qwen/Qwen3.8-27B-FP8",
                 "c3a462cc3d38cf351f3b26cbd763469eeb74ef81", "A100-80GB", 131072,
                 ("--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3",
                  "--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0}))),
}


def spec_for_app(app_name: str) -> Spec:
    """The model a deployed app serves (Gemma's when the app isn't one of ours)."""
    return next((s for s in MODELS.values() if s.app == app_name), MODELS["gemma"])


SERVER = os.environ.get("INGEST_LLM_SERVER", "gemma")   # which one this deploy is
SPEC = MODELS[SERVER]
APP_NAME, SERVED_NAME, MODEL, REVISION = SPEC.app, SPEC.served, SPEC.model, SPEC.revision
GPU, MAX_CONTEXT = SPEC.gpu, SPEC.context
CLASS_NAME = "LLMServer"
VLLM_IMAGE = "vllm/vllm-openai:v0.33.0"         # vLLM's official image (Gemma 4, Qwen 3.8)
SCALEDOWN_SECONDS = 240                         # idle time before the GPU stops
PORT = 8000

MINUTE = 60
weights = modal.Volume.from_name("ingest-llm-weights", create_if_missing=True)
compile_cache = modal.Volume.from_name("ingest-llm-cache", create_if_missing=True)
image = (modal.Image.from_registry(VLLM_IMAGE).entrypoint([])
         # Modal looks for `python`; vLLM's image has only `python3`
         .run_commands("echo PATH=$PATH; command -v python python3; python3 --version; "
                       "command -v python >/dev/null || ln -s \"$(command -v python3)\" /usr/local/bin/python")
         # the container reads the same model choice when it imports this file
         .env({"HF_HOME": "/weights/hf", "VLLM_CACHE_ROOT": "/cache/vllm",
               "INGEST_LLM_SERVER": SERVER}))
app = modal.App(APP_NAME, image=image)


@app.function(volumes={"/weights": weights}, timeout=60 * MINUTE)
def download() -> str:
    """Fetch the model into the volume (CPU only; skips files already there)."""
    from huggingface_hub import snapshot_download
    started = time.time()
    path = snapshot_download(MODEL, revision=REVISION)
    weights.commit()
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)
    return f"{MODEL} in the volume: {size / 1e9:.1f} GB, took {time.time() - started:.0f} s"


@app.cls(gpu=GPU, volumes={"/weights": weights, "/cache": compile_cache},
         scaledown_window=SCALEDOWN_SECONDS, timeout=30 * MINUTE,
         startup_timeout=25 * MINUTE, min_containers=0, max_containers=1)
@modal.concurrent(max_inputs=16)
class LLMServer:
    """One GPU at most (max_containers=1) so a burst of requests can't multiply the bill;
    vLLM answers up to 16 requests at once on it."""

    @modal.enter()
    def start(self):
        started = time.time()
        command = [
            "vllm", "serve", MODEL, "--revision", REVISION,
            "--served-model-name", SERVED_NAME, "--host", "127.0.0.1", "--port", str(PORT),
            "--max-model-len", str(MAX_CONTEXT), "--gpu-memory-utilization", "0.92",
            "--max-num-seqs", "16",
            # forms (JSON schemas) written compactly: with spaces allowed between fields,
            # Gemma once padded an agents' plan with spaces until it ran out of tokens
            "--structured-outputs-config",
            json.dumps({"backend": "xgrammar", "disable_any_whitespace": True}),
            # the agents call tools; the models think before answering only when asked to.
            # Text only: no memory set aside for pictures, sound or video (SPEC.args)
            "--enable-auto-tool-choice", *SPEC.args]
        self.process = subprocess.Popen(command, env={**os.environ, "HF_HUB_OFFLINE": "1"})
        import urllib.request
        while True:
            if self.process.poll() is not None:
                raise RuntimeError(f"vLLM stopped while starting (exit {self.process.returncode})")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except OSError:
                pass
            time.sleep(2)
        self.ready_seconds = round(time.time() - started)
        print(f"vLLM ready after {self.ready_seconds} s")
        self.client = None

    def _http(self):
        if self.client is None:
            import httpx
            self.client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORT}", timeout=25 * MINUTE)
        return self.client

    @modal.method()
    async def chat(self, body: dict) -> dict:
        """One chat completion. Returns {"status": HTTP status, "body": vLLM's JSON reply}."""
        reply = await self._http().post("/v1/chat/completions", json={**body, "model": SERVED_NAME})
        try:
            content = reply.json()
        except ValueError:
            content = {"error": {"message": reply.text[:2000]}}
        return {"status": reply.status_code, "body": content}

    @modal.method()
    async def info(self) -> dict:
        import torch
        import vllm
        return {"model": MODEL, "revision": REVISION, "served_as": SERVED_NAME,
                "gpu": torch.cuda.get_device_name(0), "vllm": vllm.__version__,
                "max_context": MAX_CONTEXT, "ready_after_seconds": self.ready_seconds}

    @modal.exit()
    def stop(self):
        self.process.terminate()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
