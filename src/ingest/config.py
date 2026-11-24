"""Settings (NFR-02): defaults < config.yaml < .env file < INGEST_* environment variables."""
from pathlib import Path
from typing import Literal

import yaml
from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Secrets such as the Qdrant Cloud API key go in .env, which is never shared with the code
    model_config = SettingsConfigDict(env_prefix="INGEST_", env_file=".env",
                                      env_file_encoding="utf-8", extra="ignore")

    # Hardware: "auto" uses an NVIDIA GPU when one is available (NFR-05)
    device: Literal["auto", "cpu", "cuda"] = "auto"

    # Cloud GPU: "modal" runs PDFs, video and picture descriptions on a Modal GPU
    gpu_backend: Literal["local", "modal"] = "local"
    modal_gpu: str = "L4"
    gpu_whisper_model: str = "large-v3-turbo"              # used instead of whisper_model on GPU
    gpu_vision_model: str = "HuggingFaceTB/SmolVLM-Instruct"  # used instead of vision_model on GPU

    # Chunking (FR-08), measured in characters
    chunk_size: int = 1000
    chunk_overlap: int = 150
    # Tables stay whole up to this many tokens, so all their rows can be compared in one chunk
    # (question, heading and table stay within the 512 tokens rerankers are trained on, and
    # MiniLM's limit); bigger ones are split by
    # rows. 0 = split tables at chunk_size characters
    table_chunk_tokens: int = 450

    # PDF (FR-04, FR-05)
    pdf_ocr: bool = True
    pdf_force_full_page_ocr: bool = False
    pdf_picture_descriptions: bool = False  # describe pictures/charts with a vision model (slow on CPU)
    pdf_page_batch: int = 40              # convert long PDFs this many pages at a time (memory)
    pdf_hidden_text_filter: bool = True   # drop white-on-white / under-3pt text
    pdf_watermark_filter: bool = True     # drop diagonal watermark text before reading
    pdf_garbage_retry: bool = True        # re-read with OCR when the text layer is garbage
    flag_instructions: bool = True        # flag chunks with "ignore previous instructions" etc.

    # Extraction runs in a worker process, so a crash (e.g. out of memory) fails one file only
    isolate_extraction: bool = True
    extract_timeout_minutes: float = 120  # stop a file that takes longer than this (0 = no limit)
    picture_prompt: str = ("Describe this image in two or three sentences. If it is a chart or "
                           "diagram, give its title, axis labels and the values shown.")

    # Word (.docx)
    docx_image_ocr: bool = True     # read text inside pictures in Word files

    # Pictures smaller than this (in pixels, either side) are skipped: logos, icons, bullets
    picture_min_px: int = 100

    # Video (FR-06, FR-07)
    whisper_model: str = "base"
    whisper_language: str | None = None  # None = auto-detect
    video_scenes: bool = True             # detect slide changes and cuts (PySceneDetect)
    video_scene_ocr: bool = True          # read on-screen text of each scene's keyframe
    video_tables: bool = True             # read tables on screen with Docling's table model
    # Equations in PDFs and video frames written as LaTeX by Docling's formula model; only
    # where a GPU is available (Modal, or a local NVIDIA card): too slow on a CPU
    formulas: bool = True
    scene_min_seconds: float = 2.0        # shorter flashes are not separate scenes
    scene_min_change: float = 2.0         # minimum change score for a cut (slides score ~3-8)
    frame_captions: bool = False          # also describe each scene with the vision model
    frame_interval_seconds: float = 10.0  # fixed sampling, used only if video_scenes is off
    vision_model: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    # Short and visual only: asked to transcribe text, small models skip the picture and get
    # the text wrong (OCR reads it better); asked for detail, they count wrongly and invent
    vision_prompt: str = ("What objects, shapes and colours can you see in this image? "
                          "Answer in one or two sentences.")

    # Embeddings (FR-10)
    embedding_model: str = "BAAI/bge-m3"  # 1024 numbers per chunk, plus sparse keyword weights
    embedding_batch_size: int = 32
    # Put before each search query. bge-m3 needs none; bge-small/base retrieve better with
    # "Represent this sentence for searching relevant passages: "
    embedding_query_prompt: str = ""
    # Exact model versions (Hugging Face commit ids), so an upstream update can't silently
    # change results; a model not listed loads its latest version
    model_revisions: dict[str, str] = {}
    # Folder with Docling's layout/table models downloaded at those revisions (the Docker
    # image has one); empty = Docling fetches its models itself
    docling_artifacts_path: str | None = None

    # Search: hybrid = meaning (dense) + keywords (bge-m3's sparse weights), merged; then the
    # best `rerank_candidates` are re-scored by a cross-encoder ("" turns reranking off)
    hybrid_search: bool = True
    bm25: bool = True                  # also rank by BM25 word weights (+2.2 nDCG@5 measured)
    reranker_model: str = "BAAI/bge-reranker-v2-m3"  # best on TAT-DQA and FinQA; lighter: MiniLM
    rerank_candidates: int = 30
    exact_match: bool = True           # ids and figures in a question (KF-26-5001, 3,663) are
                                       # also looked for as written
    passage_context_chars: int = 1200  # text added around a matched chunk (before + after)
    video_context_seconds: float = 30  # speech added before and after a matched video moment

    # Language model: answers, the question router, chat memory and agents.
    # llm_backend "gemini": Google's API (key in .env: INGEST_GEMINI_API_KEY=...,
    # aistudio.google.com/apikey). "modal": our own copy of Gemma 4 on a Modal GPU
    # (src/ingest/llm_modal.py; no daily limit, nothing leaves your Modal account); Gemini, if
    # a key is set, answers whenever it can't.
    llm_backend: Literal["gemini", "modal"] = "gemini"
    modal_llm_app: str = "ingest-llm"  # the deployed app (modal deploy src/ingest/llm_modal.py)
    gemini_api_key: SecretStr | None = None
    llm_model: str = "gemini-flash-latest"          # answers, memory snapshots, agents
    router_model: str = "gemini-flash-lite-latest"  # reads each question first: fast, cheap
    llm_temperature: float = 0.2       # randomness: 0 = always the likeliest words, 2 = wild
    llm_top_p: float = 0.95            # pick only among the words making up this much probability
    llm_max_output_tokens: int = 2048  # the longest reply
    llm_thinking_budget: int | None = None  # tokens the model may reason first (None = model's
                                            # default, 0 = none: faster, for simple calls)
    # Chat questions the router marks as needing arithmetic: the answer may think this much
    # first. Gemma answered "Karachi Port 29,670 ... 26,007 less than Hyderabad 3,663" when it
    # wrote the answer before working it out (2026-11-21); None = the model's default.
    calculation_thinking_budget: int | None = 1024
    # Tried in order while the model asked for is busy, rate-limited or out of today's quota
    llm_fallback_models: list[str] = ["gemini-3-flash-preview", "gemini-flash-lite-latest"]
    llm_timeout_seconds: float = 90    # one request; a stalled one counts as busy
    llm_retries: int = 4               # rounds through the models before giving up
    agent_max_output_tokens: int = 16384  # an agent's reply holds whole files (tables, CSV)
    answer_passages: int = 6           # passages given to the model for one answer
    # Chat memory: every snapshot_every messages, the older ones become a snapshot (entities,
    # facts, relationships, open questions) and only later ones are sent word for word
    snapshot_every: int = 6
    history_messages: int = 8          # at most this many recent messages sent word for word
    context_budget_tokens: int = 12000  # prompt size limit: snapshot + messages + passages

    # Agents (agents/): files they make go to outputs_dir/<user>
    # Their own language model, when it should differ from the chat's: agent_llm_backend
    # "gemini" or "modal" ("" = llm_backend), agent_modal_llm_app e.g. "ingest-llm-qwen"
    # ("" = modal_llm_app)
    agent_llm_backend: Literal["", "gemini", "modal"] = ""
    agent_modal_llm_app: str = ""
    outputs_dir: Path = Path("outputs")
    agent_max_steps: int = 4        # sub-tasks the planner may split a task into
    agent_max_turns: int = 10       # tool calls one agent may make before it must report
    agent_verify_rounds: int = 2    # check-and-fix rounds for each file made
    agent_sandbox: bool = True      # agents may run Python in a Modal sandbox (no network)
    agent_mcp_servers: list[dict] = []  # more MCP servers for agents: {name, command, args}
    # How the agents' steps are run: "langgraph" = a LangGraph graph (agents/graph.py), each
    # finished step saved so a stopped task can be resumed; "custom" = the plain loop
    agent_engine: Literal["langgraph", "custom"] = "langgraph"

    # Vector database tuning, applied when a collection is created (or by `ingest tune-db`);
    # each was measured in eval/vectordb_eval.py
    hnsw_m: int = 16                   # links per vector in the search graph (Qdrant's default)
    hnsw_ef_construct: int = 100       # effort spent building the graph (Qdrant's default)
    search_ef: int | None = None       # effort per search; None = Qdrant's default
    quantization: Literal["none", "int8", "binary"] = "int8"  # compressed copies in RAM
    quantization_oversampling: float = 2.0  # compressed search fetches this many times more,
                                            # then re-scores them with the full vectors
    vectors_on_disk: bool = True       # full vectors on disk: with int8 copies in RAM, a
                                       # quarter of the RAM, same accuracy (measured)
    fusion: Literal["rrf", "dbsf"] = "rrf"  # how the dense, keyword and exact rankings merge

    # Permissions: groups allowed to read each document, from rules in access_file matched
    # against its path; a search can then be limited to a user's groups
    access_file: str = "access.yaml"
    default_access: list[str] = ["everyone"]
    # Users, sessions and chats (users.py, chats.py): a SQLite file; with no users added the
    # web app has no login (one person on this machine)
    app_db: Path = Path("data/app.db")
    session_hours: float = 72

    # Server mode: a Redis job queue (e.g. redis://redis:6379/0) that `ingest worker`
    # processes take uploads from; empty = this process ingests them itself
    queue_url: str | None = None

    # Vector database (FR-11). If qdrant_url is set (e.g. Qdrant Cloud, in .env) it is used;
    # otherwise the database is kept in the local qdrant_path folder with no server.
    qdrant_url: str | None = None

    @field_validator("qdrant_url")
    @classmethod
    def _local_means_no_server(cls, value):
        """INGEST_QDRANT_URL=local (or none) uses the local folder even when .env names a
        server: PowerShell can't set a variable to "" (that deletes it)."""
        return None if value is None or value.strip().lower() in ("", "local", "none") else value
    qdrant_api_key: SecretStr | None = None  # SecretStr: never printed in logs or errors
    qdrant_path: str | None = "qdrant_data"
    collection: str = "ingestion-bge-m3-bm25"  # one per embedding model and vector layout

    # Output
    uploads_dir: Path = Path("data/uploads")  # where files uploaded in the web app are kept
    extracts_dir: Path = Path("extracted")    # saved extractions, reused instead of re-reading
    reports_dir: Path = Path("reports")
    exports_dir: Path = Path("exports")
    logs_dir: Path = Path("logs")
    log_level: str = "INFO"

    def use_cuda(self) -> bool:
        """True when an NVIDIA GPU should be used (NFR-05)."""
        if self.device != "auto":
            return self.device == "cuda"
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False


def load_settings(config_path: str | Path | None = None, **overrides) -> Settings:
    """Load config.yaml (if present), then let .env and INGEST_* env vars override it."""
    file_values: dict = {}
    path = Path(config_path) if config_path else Path("config.yaml")
    if path.exists():
        file_values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    elif config_path:
        raise FileNotFoundError(f"Config file not found: {path}")

    # Env vars beat the file: build from env first, keep only fields not set there
    from_env = Settings()
    env_set = from_env.model_fields_set
    merged = {k: v for k, v in file_values.items() if k not in env_set}
    merged.update({k: getattr(from_env, k) for k in env_set})
    merged.update(overrides)
    return Settings.model_validate(merged)
