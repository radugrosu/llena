from __future__ import annotations

import random
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from mm.types import VQASample


class SyntheticVQADataset(Dataset):
    """Minimal synthetic VQA dataset for stage-0 smoke tests.

    Generates simple colored images with a colored rectangle and asks a question
    whose answer is deterministic (so loss should decrease).
    """

    def __init__(self, num_samples: int = 64, image_size: int = 224, seed: int = 42):
        self.num_samples = int(num_samples)
        self.image_size = int(image_size)
        self.rng = random.Random(seed)

        self.colors: list[tuple[str, tuple[int, int, int]]] = [
            ("red", (220, 40, 40)),
            ("green", (40, 180, 70)),
            ("blue", (40, 90, 220)),
            ("yellow", (220, 200, 40)),
            ("purple", (160, 60, 200)),
            ("orange", (230, 140, 40)),
        ]

    def __len__(self) -> int:
        return self.num_samples

    def _make_image(self, color_rgb: tuple[int, int, int]) -> Image.Image:
        img = Image.new("RGB", (self.image_size, self.image_size), (245, 245, 245))
        draw = ImageDraw.Draw(img)

        # Draw a filled rectangle of the chosen color (center-ish)
        margin = self.image_size // 6
        x0, y0 = margin, margin
        x1, y1 = self.image_size - margin, self.image_size - margin
        draw.rectangle([x0, y0, x1, y1], fill=color_rgb)

        return img

    def __getitem__(self, idx: int) -> VQASample:
        # Pick a color deterministically-ish but still varied
        name, rgb = self.colors[idx % len(self.colors)]
        img = self._make_image(rgb)

        question = "What is the color of the rectangle?"
        answer = name

        return {
            "image": img,
            "question": question,
            "answer": answer,
        }
