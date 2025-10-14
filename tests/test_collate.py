"""Pruebas unitarias para el collate de ViT.

Estas pruebas validan que:
- El collate devuelve tensores con las claves esperadas.
- El tamaño del batch en `pixel_values` y `labels` coincide con la
  cantidad de ejemplos del lote.
"""

from __future__ import annotations

from typing import List

from PIL import Image
from transformers import AutoImageProcessor

from plant_disease.training.train import ViTCollator


def _make_dummy_batch(n: int = 1) -> List[dict]:
    """Crea un batch sintético de `n` imágenes 224x224 RGB con etiqueta 0."""
    img = Image.new("RGB", (224, 224), (255, 255, 255))
    return [{"image": img, "label": 0} for _ in range(n)]


def test_vitcollator_basic():
    """El collate debe generar `pixel_values` y `labels` con batch size 1."""
    processor = AutoImageProcessor.from_pretrained(
        "google/vit-base-patch16-224-in21k", use_fast=True
    )
    collate = ViTCollator(processor)

    batch = _make_dummy_batch(n=1)
    out = collate(batch)

    assert "pixel_values" in out and "labels" in out
    assert out["pixel_values"].shape[0] == 1
    assert out["labels"].shape[0] == 1


def test_vitcollator_batch_size_two():
    """El collate debe respetar batch size 2."""
    processor = AutoImageProcessor.from_pretrained(
        "google/vit-base-patch16-224-in21k", use_fast=True
    )
    collate = ViTCollator(processor)

    batch = _make_dummy_batch(n=2)
    out = collate(batch)

    assert out["pixel_values"].shape[0] == 2
    assert out["labels"].shape[0] == 2