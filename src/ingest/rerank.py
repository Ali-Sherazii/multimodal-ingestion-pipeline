"""Rerank: re-score the best search candidates with a cross-encoder.

The embedding models compare a question and a chunk only through their vectors, made
separately. A cross-encoder reads the question and the chunk together, which ranks better
but is too slow to run on every chunk, so it only re-orders the top `rerank_candidates`.
"""
from collections.abc import Callable
from functools import lru_cache


@lru_cache(maxsize=2)
def token_counter(model_name: str, revision: str | None = None) -> Callable[[str], int]:
    """Counts text in this model's tokens. Loads only its tokenizer (about 1 MB), not the
    model: table chunks are sized by it (see chunk_segments)."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    tokenizer.model_max_length = 10**9  # only counting: no warning for long text
    return lambda text: len(tokenizer.tokenize(text))


def _text(payload: dict) -> str:
    heading = payload.get("section_heading")
    return f"{heading}\n{payload['content']}" if heading else payload["content"]


class Reranker:
    def __init__(self, model_name: str, device: str = "auto", revision: str | None = None):
        from sentence_transformers import CrossEncoder
        self.model_name = model_name
        self.model = CrossEncoder(model_name, revision=revision,
                                  device=None if device == "auto" else device)
        if self.model.device.type == "cuda":  # half precision: several times faster, same order
            self.model.model.half()

    def rerank(self, query: str, hits: list) -> list:
        """The hits re-ordered by the cross-encoder; each hit's score becomes its score."""
        if not hits:
            return hits
        scores = self.model.predict([(query, _text(h.payload)) for h in hits], batch_size=32,
                                    show_progress_bar=False)
        for hit, score in zip(hits, scores, strict=True):
            hit.score = float(score)
        return sorted(hits, key=lambda h: h.score, reverse=True)
