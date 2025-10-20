"""Evaluación del modelo ViT en un split del dataset.

Este script carga un modelo entrenado (formato `save_pretrained`) y evalúa
su rendimiento sobre un split del dataset de Hugging Face, calculando:

- Accuracy global
- Pérdidas promedio (cross-entropy del modelo)
- Matriz de confusión (si `scikit-learn` está disponible)
- Reporte por clase (precision/recall/F1, si `scikit-learn` está disponible)

Resultados:
- Guarda `evaluation.json` con métricas globales.
- Guarda `confusion_matrix.csv` (si `sklearn` está instalado).
- Opcionalmente imprime un resumen en consola.

Ejemplo:
    python -m plant_disease.evaluation.evaluate \
        --model_dir runs/vit-gvj/final \
        --hf_path GVJahnavi/PlantVillage_dataset \
        --split test \
        --batch_size 32 \
        --out outputs/eval_vit_gvj
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from datasets import ClassLabel, Dataset, load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, ViTForImageClassification

# Import local utilidades de carga si las necesitas (opcional):
# from plant_disease.data.datasets import load_generic_from_hub


# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------
def ensure_dir(path: str) -> None:
    """Crea un directorio si no existe."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _maybe_load_class_names_from_model(model_dir: str) -> list[str] | None:
    """Intenta cargar nombres de clases desde el directorio del modelo.

    Busca `class_names.json` o `labels.json`. Si no existen, retorna None.
    """
    for fname in ("class_names.json", "labels.json"):
        fpath = os.path.join(model_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, encoding="utf-8") as f:
                return json.load(f)
    return None


def _load_class_names_from_hub(hf_path: str, split: str) -> list[str]:
    """Obtiene nombres de clases desde el dataset de Hugging Face."""
    raw = load_dataset(hf_path)
    split_name = split if split in raw else list(raw.keys())[0]
    feat = raw[split_name].features.get("label")
    if isinstance(feat, ClassLabel):
        return list(feat.names)
    # Fallback: generar nombres genéricos
    labels = raw[split_name]["label"]
    n = int(max(labels)) + 1 if len(labels) else 0
    return [f"class_{i}" for i in range(n)]


def _load_split(
    hf_path: str,
    split: str,
    image_column: str = "image",
    label_column: str = "label",
    cache_dir: str | None = None,
) -> Dataset:
    """Carga un split del dataset y normaliza columnas a `image` y `label`."""
    dset = load_dataset(hf_path, split=split, cache_dir=cache_dir)
    # Forzar tipos/renombres mínimos (evitamos dependencia cruzada con training)
    from datasets import Image as HFImage

    if not isinstance(dset.features.get(image_column), HFImage):
        dset = dset.cast_column(image_column, HFImage())

    rename: dict[str, str] = {}
    if image_column != "image":
        rename[image_column] = "image"
    if label_column != "label":
        rename[label_column] = "label"
    if rename:
        dset = dset.rename_columns(rename)

    keep = {"image", "label"}
    drop = [c for c in dset.column_names if c not in keep]
    if drop:
        dset = dset.remove_columns(drop)

    return dset


# ---------------------------------------------------------------------
# Collate (idéntico en espíritu al de entrenamiento)
# ---------------------------------------------------------------------
@dataclass
class ViTCollator:
    """Convierte ejemplos del dataset en tensores listos para ViT."""

    processor: AutoImageProcessor

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images: list[Image.Image] = []
        labels: list[int] = []

        for item in batch:
            img = item["image"]
            if isinstance(img, Image.Image):
                pil = img.convert("RGB")
            else:
                pil = Image.fromarray(np.asarray(img)).convert("RGB")
            images.append(pil)
            labels.append(int(item["label"]))

        inputs = self.processor(images=images, return_tensors="pt")
        inputs["labels"] = torch.tensor(labels, dtype=torch.long)
        return inputs


# ---------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: ViTForImageClassification,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Evalúa el modelo en un DataLoader.

    Retorna:
        loss_promedio, accuracy, y_true(np.ndarray), y_pred(np.ndarray)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    seen = 0

    all_true: list[int] = []
    all_pred: list[int] = []

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss
        logits = out.logits
        preds = logits.argmax(dim=-1)

        bs = batch["labels"].size(0)
        total_loss += float(loss.item()) * bs
        correct += int((preds == batch["labels"]).sum().item())
        seen += bs

        all_true.extend(batch["labels"].tolist())
        all_pred.extend(preds.tolist())

    avg_loss = total_loss / max(1, seen)
    acc = correct / max(1, seen)
    return avg_loss, acc, np.asarray(all_true), np.asarray(all_pred)


def main() -> None:
    """Punto de entrada: evalúa un modelo ViT en un split del dataset."""
    parser = argparse.ArgumentParser(description="Evaluar modelo ViT")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="runs/vit-gvj/final",
        help="Directorio del modelo (save_pretrained).",
    )
    parser.add_argument(
        "--hf_path",
        type=str,
        default="GVJahnavi/PlantVillage_dataset",
        help="Ruta/dataset de Hugging Face.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split a evaluar (p. ej. test, validation, train).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Tamaño de batch para evaluación.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Workers del DataLoader (0 en Windows es lo más estable).",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directorio de caché para datasets (opcional).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/eval",
        help="Directorio donde guardar métricas y artefactos.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.out)

    # Cargar modelo y preprocesador
    processor = AutoImageProcessor.from_pretrained(args.model_dir)
    model = ViTForImageClassification.from_pretrained(args.model_dir)
    model.to(device)

    # Cargar nombres de clases (modelo -> dataset)
    class_names = _maybe_load_class_names_from_model(args.model_dir)
    if class_names is None:
        class_names = _load_class_names_from_hub(args.hf_path, args.split)

    # Dataset + DataLoader
    dset = _load_split(
        hf_path=args.hf_path,
        split=args.split,
        image_column="image",
        label_column="label",
        cache_dir=args.cache_dir,
    )
    collate = ViTCollator(processor)
    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )

    # Evaluar
    loss, acc, y_true, y_pred = evaluate(model, loader, device)

    # Métricas adicionales con sklearn (si está disponible)
    metrics: dict[str, Any] = {
        "loss": float(loss),
        "accuracy": float(acc),
        "num_samples": int(len(y_true)),
    }

    try:
        from sklearn.metrics import (  # type: ignore
            classification_report,
            confusion_matrix,
        )

        report = classification_report(
            y_true,
            y_pred,
            target_names=class_names if len(class_names) > 0 else None,
            output_dict=True,
            zero_division=0,
        )
        metrics["classification_report"] = report

        cm = confusion_matrix(y_true, y_pred)
        # Guardar matriz de confusión como CSV
        cm_path = os.path.join(args.out, "confusion_matrix.csv")
        np.savetxt(cm_path, cm, fmt="%d", delimiter=",")
        print(f"Guardada matriz de confusión en: {cm_path}")
    except Exception:
        # sklearn no está instalado: guardamos solo lo básico
        metrics["note"] = (
            "Instala scikit-learn para reporte por clase y matriz de confusión."
        )

    # Guardar métricas
    out_json = os.path.join(args.out, "evaluation.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"✅ Evaluación guardada en: {out_json}")
    print(f"Accuracy: {acc:.4f}   Loss: {loss:.4f}")


if __name__ == "__main__":
    main()
