"""Exportación de un modelo ViT (Hugging Face) a TorchScript.

Este script carga un modelo entrenado (guardado con ``save_pretrained``)
y lo exporta a TorchScript. Primero intenta `torch.jit.script` y, si falla
(por partes no-scriptables), hace fallback automático a `torch.jit.trace`.

Ejemplos
--------
Exportar el modelo final entrenado:

    python -m plant_disease.inference.export_torchscript \
        --model_dir runs/vit-gvj/final \
        --out exports/vit_gvj.ts

Opciones
--------
- --model_dir:  Directorio con artefactos HF (config.json, pesos, etc.).
- --out:        Ruta del archivo .ts a generar.
- --image_size: Tamaño (H, W) esperado por el backbone (por defecto, 224).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoImageProcessor

from plant_disease.models.vit import VitClassifier, VitConfig


def _parse_size(size: int | str) -> Tuple[int, int]:
    """Devuelve (H, W) a partir de un entero o 'HxW'."""
    if isinstance(size, int):
        return size, size
    if "x" in size.lower():
        h, w = size.lower().split("x")
        return int(h), int(w)
    v = int(size)
    return v, v


def main() -> None:
    """Punto de entrada: exporta el modelo a TorchScript."""
    parser = argparse.ArgumentParser(description="Exportar modelo a TorchScript")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directorio del modelo (save_pretrained).",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Ruta de salida del archivo .ts",
    )
    parser.add_argument(
        "--image_size",
        type=str,
        default="224",
        help="Tamaño de imagen (entero o 'HxW'), p. ej. 224 o 224x224.",
    )
    args = parser.parse_args()

    # Cargar processor para inferir tamaño si está disponible.
    processor = AutoImageProcessor.from_pretrained(args.model_dir)
    if args.image_size == "224" and hasattr(processor, "size"):
        size = processor.size
        if isinstance(size, dict) and "height" in size and "width" in size:
            h, w = size["height"], size["width"]
        else:
            h = w = 224
    else:
        h, w = _parse_size(args.image_size)

    # Cargar el backbone mediante el wrapper VitClassifier.
    # Nota: VitConfig usa model_name como path del directorio HF.
    config = VitConfig(model_name=args.model_dir, num_labels=1)  # num_labels no se usa aquí
    clf = VitClassifier(config)
    model = clf.model
    model.eval()

    # Dummy input para trace (NCHW).
    dummy = torch.randn(1, 3, h, w)

    # Intentar script primero; si falla, fallback a trace.
    try:
        scripted = torch.jit.script(model)
        scripted_type = "script"
    except Exception:
        scripted = torch.jit.trace(model, (dummy,), strict=False)
        scripted_type = "trace"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    scripted.save(args.out)
    print(f"✅ Modelo exportado a TorchScript ({scripted_type}) en: {args.out}")


if __name__ == "__main__":
    main()
