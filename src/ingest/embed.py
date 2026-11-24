"""Embed (FR-10): turn chunk text into vectors with a configurable local model.

Each text gets a dense vector (its meaning) and, with bge-m3, a sparse one: a weight for each
of its words, so exact terms (names, codes, numbers) can be matched too. Both come from one
pass of one model.
"""
import os
from typing import Protocol

from ingest.models import Chunk

Sparse = dict[int, float]  # token id -> weight


class Embedder(Protocol):
    model_name: str
    dimension: int
    has_sparse: bool

    def embed_documents(self, texts: list[str]) -> tuple[list[list[float]], list[Sparse] | None]: ...
    def embed_query(self, text: str) -> tuple[list[float], Sparse | None]: ...


class _LexicalWeights:
    """bge-m3's sparse (lexical) weights, as its authors' code computes them: a linear layer
    (sparse_linear.pt, shipped with the model) over the token states the dense pass already
    produces; weight = relu(w·h + b); a word that appears twice keeps its higher weight;
    special tokens are left out."""

    def __init__(self, model_name: str, revision: str | None, tokenizer, device):
        import torch
        from huggingface_hub import hf_hub_download
        state = torch.load(hf_hub_download(model_name, "sparse_linear.pt", revision=revision),
                           map_location="cpu", weights_only=True)
        self.linear = torch.nn.Linear(state["weight"].shape[1], 1)
        self.linear.load_state_dict(state)
        self.linear.to(device).eval()
        self.device = device
        self.skip = {i for i in (tokenizer.cls_token_id, tokenizer.eos_token_id,
                                 tokenizer.pad_token_id, tokenizer.unk_token_id) if i is not None}

    @staticmethod
    def available(model_name: str, revision: str | None) -> bool:
        from huggingface_hub import hf_hub_download
        try:
            hf_hub_download(model_name, "sparse_linear.pt", revision=revision)
            return True
        except Exception:  # not a bge-m3-style model (no such file), or offline without it
            return False

    def __call__(self, row: dict) -> Sparse:
        import torch
        with torch.no_grad():
            weights = torch.relu(self.linear(row["token_embeddings"].to(self.device))).squeeze(-1)
        out: Sparse = {}
        for token, weight, keep in zip(row["input_ids"].tolist(), weights.tolist(),
                                       row["attention_mask"].tolist()):
            if keep and token not in self.skip and weight > 0:
                out[token] = max(out.get(token, 0.0), weight)
        return out


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 32,
                 query_prompt: str = "", revision: str | None = None, sparse: bool = False):
        # Models published only as pytorch_model.bin (bge-m3) otherwise make transformers start
        # a background download of a converted copy (2.3 GB for bge-m3), and the program can't
        # exit until it finishes
        os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.batch_size = batch_size
        self.query_prompt = query_prompt  # BGE: "Represent this sentence for searching …"
        self.model = SentenceTransformer(model_name, revision=revision,
                                         device=None if device == "auto" else device)
        get_dim = (getattr(self.model, "get_embedding_dimension", None)  # sentence-transformers 6+
                   or self.model.get_sentence_embedding_dimension)
        self.dimension = get_dim()
        self.lexical = (_LexicalWeights(model_name, revision, self.model.tokenizer, self.model.device)
                        if sparse and _LexicalWeights.available(model_name, revision) else None)
        self.has_sparse = self.lexical is not None

    def _dense_and_sparse(self, texts: list[str], prompt: str = ""):
        """One forward pass per batch, keeping both the sentence vector and the token states.

        Not encode(output_value=None): in sentence-transformers 6.1 it splits string-valued
        features ("task") per item, which crashes past the 8th text of a batch.
        """
        import torch
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # less padding
        dense: list = [None] * len(texts)
        sparse: list = [None] * len(texts)
        for start in range(0, len(order), self.batch_size):
            ids = order[start:start + self.batch_size]
            # preprocess() replaced tokenize() in sentence-transformers 6 (same output)
            prepare = getattr(self.model, "preprocess", None) or self.model.tokenize
            features = prepare([prompt + texts[i] for i in ids])
            features = {k: v.to(self.model.device) for k, v in features.items()
                        if isinstance(v, torch.Tensor)}
            with torch.no_grad():
                out = self.model(features)
            for row, i in enumerate(ids):
                dense[i] = torch.nn.functional.normalize(out["sentence_embedding"][row], dim=0).tolist()
                sparse[i] = self.lexical({name: out[name][row] for name in
                                          ("token_embeddings", "input_ids", "attention_mask")})
        return dense, sparse

    def embed_documents(self, texts: list[str]) -> tuple[list[list[float]], list[Sparse] | None]:
        if self.lexical is not None:
            return self._dense_and_sparse(texts)
        vectors = self.model.encode_document(texts, batch_size=self.batch_size,
                                             normalize_embeddings=True)
        return vectors.tolist(), None

    def embed_query(self, text: str) -> tuple[list[float], Sparse | None]:
        # Short questions are matched against long passages: the instruction tells the
        # model which side this text is on. Documents are embedded without it.
        if self.lexical is not None:
            dense, sparse = self._dense_and_sparse([text], self.query_prompt)
            return dense[0], sparse[0]
        return self.model.encode_query(text, prompt=self.query_prompt or None,
                                       normalize_embeddings=True).tolist(), None


class Bm25:
    """BM25 keyword weights (fastembed's Qdrant/bm25): words stemmed, stop words dropped, each
    word weighted by how often it occurs, adjusted for the chunk's length. Qdrant adds how rare
    each word is across the collection (IDF). Complements bge-m3's own word weights: measured
    +2.2 nDCG@5 on TAT-DQA (eval/vectordb_eval.py). No model to run, so it is fast on a CPU."""
    model_name = "Qdrant/bm25"

    def __init__(self):
        from pathlib import Path

        from fastembed import SparseTextEmbedding
        cache = os.environ.get("FASTEMBED_CACHE_PATH") or str(Path.home() / ".cache" / "fastembed")
        self.model = SparseTextEmbedding(self.model_name, cache_dir=cache)

    @staticmethod
    def _weights(e) -> Sparse:
        return {int(i): float(v) for i, v in zip(e.indices, e.values)}

    def documents(self, texts: list[str]) -> list[Sparse]:
        return [self._weights(e) for e in self.model.embed(texts, batch_size=256)]

    def query(self, text: str) -> Sparse:
        return self._weights(next(iter(self.model.query_embed(text))))


def text_for_embedding(chunk: Chunk) -> str:
    # The heading adds context a short chunk lacks ("Sick Leave" + "10 days, paid")
    if chunk.section_heading:
        return f"{chunk.section_heading}\n{chunk.content}"
    return chunk.content


def embed_chunks(chunks: list[Chunk], embedder: Embedder, bm25: Bm25 | None = None) -> None:
    texts = [text_for_embedding(c) for c in chunks]
    dense, sparse = embedder.embed_documents(texts)
    words = bm25.documents(texts) if bm25 is not None else None
    for i, (chunk, vector) in enumerate(zip(chunks, dense, strict=True)):
        chunk.embedding = vector
        chunk.sparse = sparse[i] if sparse else None
        chunk.bm25 = words[i] if words else None
