"""Exportación de un modelo ViT (Hugging Face) a formato ONNX.

Este script carga un modelo entrenado (guardado con ``save_pretrained``)
y lo exporta a ONNX con una entrada única: ``pixel_values`` (B, 3, 224, 224).

Ejemplos
--------
Exportar el modelo final entrenado:

    python -m plant_disease.inference.export_onnx \
        --model_dir runs/vit-gvj/final \
        --out exports/vit_gvj.onnx

Opciones
--------
- --model_dir:  Directorio con artefactos HF (config.json, pesos, etc.).
- --out:        Ruta del archivo .onnx a generar.
- --opset:      Versión de ONNX opset (por defecto, 17).
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
    """Punto de entrada: exporta el modelo a ONNX."""
    parser = argparse.ArgumentParser(description="Exportar modelo a ONNX")
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
        help="Ruta de salida del archivo .onnx",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="Versión de opset de ONNX (por defecto, 17).",
    )
    parser.add_argument(
        "--image_size",
        type=str,
        default="224",
        help="Tamaño de imagen (entero o 'HxW'), p. ej. 224 o 224x224.",
    )
    args = parser.parse_args()

    # Carga del processor para inferir parámetros (normalización, tamaño, etc.)
    processor = AutoImageProcessor.from_pretrained(args.model_dir)
    # Intentamos obtener el tamaño desde el processor si no se especifica
    if args.image_size == "224" and hasattr(processor, "size"):
        size = processor.size
        if isinstance(size, dict) and "height" in size and "width" in size:
            h, w = size["height"], size["width"]
        else:
            h = w = 224
    else:
        h, w = _parse_size(args.image_size)

    # Carga del backbone mediante el wrapper (mismos pesos del directorio)
    # Nota: VitClassifier encapsula un AutoModelForImageClassification.
    # Aquí solo necesitamos su .model para exportar el forward estándar.
    # El num_labels se infiere de la carpeta (config.json).
    # Para construir VitConfig, usamos el mismo model_name que el directorio.
    config = VitConfig(model_name=args.model_dir, num_labels=1)  # num_labels no se usa
    clf = VitClassifier(config)
    model = clf.model
    model.eval()

    # Entrada dummy con el tamaño esperado por el modelo.
    dummy = torch.randn(1, 3, h, w)

    # Asegurar directorio de salida
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Exportación: la firma usa el forward del modelo HF
    torch.onnx.export(
        model,
        (dummy,),
        args.out,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
    )
    print(f"✅ Modelo exportado a ONNX en: {args.out}")


if __name__ == "__main__":
    main()