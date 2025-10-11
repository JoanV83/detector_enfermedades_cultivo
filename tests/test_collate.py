f"""Pruebas unitarias del collator ViTCollator.

Este módulo valida que el collator:
- Convierta correctamente imágenes PIL a tensores listos para ViT.
- Devuelva las claves esperadas ("pixel_values", "labels").
- Respete el tamaño del batch tanto para 1 como para N ejemplos.
"""

from __future__ import annotations

# 1) Importaciones estándar de Python
from typing import Any, Dict, List

# 2) Importaciones de terceros
from PIL import Image
from transformers import AutoImageProcessor

# 3) Importaciones locales
from plant_disease.training.train import ViTCollator


def _dummy_image(rgb: int = 255, size: int = 224) -> Image.Image:
    """Crea una imagen PIL RGB cuadrada de color sólido.

    Args:
        rgb: Intensidad de color (0–255) para R=G=B.
        size: Ancho y alto (en píxeles).

    Returns:
        Imagen PIL en modo "RGB".
    """
    return Image.new("RGB", (size, size), (rgb, rgb, rgb))


def test_vitcollator_single_item() -> None:
    """El collator debe funcionar con un solo elemento en el batch."""
    processor = AutoImageProcessor.from_pretrained(
        "google/vit-base-patch16-224-in21k", use_fast=True
    )
    collate = ViTCollator(processor)

    batch: List[Dict[str, Any]] = [{"image": _dummy_image(), "label": 0}]
    out = collate(batch)

    assert "pixel_values" in out and "labels" in out
    assert out["pixel_values"].shape[0] == 1  # batch size
    assert out["labels"].shape[0] == 1
    # Los labels deben ser enteros tipo long
    assert out["labels"].dtype.name in ("int64", "long")


def test_vitcollator_multiple_items() -> None:
    """El collator debe respetar el tamaño del batch > 1."""
    processor = AutoImageProcessor.from_pretrained(
        "google/vit-base-patch16-224-in21k", use_fast=True
    )
    collate = ViTCollator(processor)

    batch: List[Dict[str, Any]] = [
        {"image": _dummy_image(rgb=255), "label": 0},
        {"image": _dummy_image(rgb=128), "label": 1},
        {"image": _dummy_image(rgb=64), "label": 0},
    ]
    out = collate(batch)

    assert "pixel_values" in out and "labels" in out
    assert out["pixel_values"].shape[0] == 3
    assert out["labels"].shape[0] == 3