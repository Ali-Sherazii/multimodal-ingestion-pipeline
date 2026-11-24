"""Download every model the CPU image needs, at the exact versions pinned in config.yaml
(model_revisions), so the container runs with HF_HUB_OFFLINE=1 and nothing can change
underneath it. Run at image build time (Dockerfile):

  - BAAI/bge-m3 (dense + sparse head), the configured reranker and Whisper base -> Hugging Face cache
  - Docling's layout and table models -> INGEST_DOCLING_ARTIFACTS_PATH, in the folder names
    Docling expects, plus the OCR models Docling's own downloader provides
  - RapidOCR's models for video frames and Word pictures (inside its package)

GPU-only models (Whisper large-v3-turbo, SmolVLM, CodeFormulaV2) live in the Modal image.
"""
import os
from pathlib import Path

from huggingface_hub import snapshot_download

from ingest.config import load_settings

PATTERNS = {  # only the files these models are loaded from (bge-m3 also ships a 2.3 GB ONNX copy)
    "BAAI/bge-m3": ["*.json", "pytorch_model.bin", "sentencepiece.bpe.model", "1_Pooling/*",
                    "sparse_linear.pt"],
    "Systran/faster-whisper-base": ["config.json", "model.bin", "tokenizer.json",
                                    "vocabulary.*", "preprocessor_config.json"],
}
RERANKERS = {  # the one config.yaml names is added to the image
    "BAAI/bge-reranker-v2-m3": ["*.json", "model.safetensors", "sentencepiece.bpe.model"],
    "cross-encoder/ms-marco-MiniLM-L-6-v2": ["*.json", "*.txt", "model.safetensors"],
}
DOCLING = {"docling-project/docling-layout-heron": None,
           "docling-project/docling-models": ["model_artifacts/tableformer/*"]}


def main() -> None:
    settings = load_settings()
    pins = settings.model_revisions
    models = dict(PATTERNS)
    if settings.reranker_model:
        models[settings.reranker_model] = RERANKERS[settings.reranker_model]
    for repo, patterns in models.items():
        print(f"{repo} @ {pins[repo][:10]}", flush=True)
        snapshot_download(repo, revision=pins[repo], allow_patterns=patterns)

    artifacts = Path(os.environ.get("INGEST_DOCLING_ARTIFACTS_PATH")
                     or settings.docling_artifacts_path or "/models/docling")
    for repo, patterns in DOCLING.items():
        print(f"{repo} @ {pins[repo][:10]} -> {artifacts}", flush=True)
        snapshot_download(repo, revision=pins[repo], allow_patterns=patterns,
                          local_dir=artifacts / repo.replace("/", "--"))
    from docling.utils.model_downloader import download_models
    download_models(output_dir=artifacts, with_layout=False, with_tableformer=False,
                    with_code_formula=False, with_picture_classifier=False, with_rapidocr=True)

    from ingest.ocr import _engine
    _engine()  # fetches RapidOCR's own models into its package once
    if settings.bm25:
        from ingest.embed import Bm25
        Bm25()  # its word lists into FASTEMBED_CACHE_PATH (set in the Dockerfile)
    print("models ready", flush=True)


if __name__ == "__main__":
    main()
