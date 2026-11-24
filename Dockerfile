# CPU image for server mode: the web app (`ingest web`) and the queue workers (`ingest worker`),
# run together by docker-compose.yml. Extraction, chunking, embedding and storage happen here,
# not on anyone's laptop; slow steps can still go to the Modal GPU (INGEST_GPU_BACKEND=modal,
# with MODAL_TOKEN_ID / MODAL_TOKEN_SECRET).
#
# Reproducible: base image, libraries (requirements.lock) and models (model_revisions in
# config.yaml) are all pinned to exact versions, and the models are baked in, so the
# container never downloads anything at run time.
FROM python:3.12.15-slim-bookworm

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/hf HF_HUB_DISABLE_SYMLINKS_WARNING=1 DISABLE_SAFETENSORS_CONVERSION=1 \
    INGEST_DOCLING_ARTIFACTS_PATH=/models/docling FASTEMBED_CACHE_PATH=/models/fastembed

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# PyTorch's CPU-only build (the default one bundles 2+ GB of CUDA libraries a CPU can't use)
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 torchvision==0.29.1
COPY requirements.lock .
RUN pip install -r requirements.lock

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-deps .

COPY config.yaml access.yaml ./
COPY scripts/download_models.py scripts/
RUN python scripts/download_models.py
ENV HF_HUB_OFFLINE=1

RUN useradd --create-home ingest && mkdir -p /data && chown ingest /data
USER ingest
EXPOSE 8000
CMD ["ingest", "web", "--host", "0.0.0.0", "--no-browser"]
