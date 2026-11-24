"""Image captioning with a small vision-language model (SmolVLM by default).

Shared by video frame captions (FR-07) and picture descriptions in PDFs and Word files.
Loaded with transformers, which fetches only the files the model needs. 
"""
from functools import lru_cache

from PIL import Image

from ingest.config import Settings


class ImageCaptioner:
    def __init__(self, model_id: str, cuda: bool, revision: str | None = None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.device = "cuda" if cuda else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, revision=revision,
            dtype=torch.float16 if cuda else torch.float32).to(self.device)

    def caption(self, image: Image.Image, prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[image.convert("RGB")],
                                return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        new_tokens = output[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


@lru_cache(maxsize=1)
def _captioner(model_id: str, cuda: bool, revision: str | None) -> ImageCaptioner:
    return ImageCaptioner(model_id, cuda, revision)


def get_captioner(settings: Settings) -> ImageCaptioner:
    """One model instance per run, whether it is captioning video frames or PDF pictures."""
    return _captioner(settings.vision_model, settings.use_cuda(),
                      settings.model_revisions.get(settings.vision_model))
