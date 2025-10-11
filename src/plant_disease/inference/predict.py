"""Inferencia con un modelo ViT entrenado para PlantVillage.

Provee utilidades para cargar el modelo/processor desde un directorio de
artefactos (``runs/...``) y predecir el top-k de clases para una imagen.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from datasets import ClassLabel, load_dataset
from transformers import AutoImageProcessor, ViTForImageClassification

DEFAULT_MODEL_DIR = "runs/vit-gvj/final"
DEFAULT_DATASET = "GVJahnavi/PlantVillage_dataset"


def _load_class_names_from_hub(hf_path: str = DEFAULT_DATASET) -> List[str]:
    """Obtiene nombres de clases desde el dataset del Hub."""
    raw = load_dataset(hf_path)
    split = "train" if "train" in raw else list(raw.keys())[0]
    feat = raw[split].features.get("label")
    if isinstance(feat, ClassLabel):
        return feat.names

    labels = raw[split]["label"]
    n = int(max(labels)) + 1 if len(labels) else 0
    return [f"class_{i}" for i in range(n)]


def _load_class_names(model_dir: str) -> List[str]:
    """Lee ``labels.json`` del directorio del modelo o cae al Hub."""
    labels_json = os.path.join(model_dir, "labels.json")
    if os.path.exists(labels_json):
        with open(labels_json, "r", encoding="utf-8") as f:
            return json.load(f)
    return _load_class_names_from_hub()


def load_model(
    model_dir: str = DEFAULT_MODEL_DIR,
    device: str | None = None,
):
    """Carga processor, modelo y nombres de clase desde ``model_dir``.

    Inyecta ``id2label`` y ``label2id`` en la config para asegurar que los
    índices de salida correspondan a las clases correctas.

    Args:
        model_dir: Directorio con los artefactos (``config.json``, pesos, etc.).
        device: ``"cuda"`` o ``"cpu"``. Por defecto, autodetección.

    Returns:
        (model, processor, class_names, torch.device)
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    processor = AutoImageProcessor.from_pretrained(model_dir, use_fast=True)
    model = ViTForImageClassification.from_pretrained(model_dir)
    class_names = _load_class_names(model_dir)

    id2label = {i: name for i, name in enumerate(class_names)}
    label2id = {name: i for i, name in enumerate(class_names)}
    model.config.id2label = id2label
    model.config.label2id = label2id

    model.to(device)
    model.eval()
    return model, processor, class_names, device


@torch.no_grad()
def predict(
    image_path: str,
    model_dir: str = DEFAULT_MODEL_DIR,
    topk: int = 5,
) -> List[Tuple[str, float]]:
    """Predice el top-k de clases para una imagen.

    Args:
        image_path: Ruta a la imagen (cualquier formato soportado por PIL).
        model_dir: Directorio del modelo entrenado.
        topk: Número de resultados a devolver ordenados por probabilidad.

    Returns:
        Lista de tuplas (nombre_clase, probabilidad).
    """
    model, processor, class_names, device = load_model(model_dir)
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    top_idx = np.argsort(probs)[::-1][:topk]

    return [
        (class_names[i] if i < len(class_names) else f"class_{i}", float(probs[i]))
        for i in top_idx
    ]


def main() -> None:
    """CLI simple para predecir una imagen."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Ruta a la imagen a predecir.")
    ap.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    preds = predict(args.image, model_dir=args.model_dir, topk=args.topk)
    print("Top-K predicciones:")
    for name, p in preds:
        print(f"{name:40s}  {p:.4f}")


if __name__ == "__main__":
    main()
