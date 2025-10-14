"""Guardar imágenes de muestra del dataset en data/test_images/.

Este script descarga unas cuantas imágenes del dataset de Hugging Face
(GVJahnavi/PlantVillage_dataset por defecto) y las guarda en
`data/test_images/` para probar rápidamente la app de Streamlit.

Uso:
    python scripts/save_samples.py
    # Opcionalmente:
    python scripts/save_samples.py --n-train 5 --n-test 5 --out data/test_images
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

from datasets import ClassLabel, load_dataset
from PIL import Image


def _as_pil(x) -> Image.Image:
    """Asegura que x sea PIL.Image (soporta PIL o np.ndarray)."""
    if isinstance(x, Image.Image):
        return x
    return Image.fromarray(x)


def _get_label_name(dset_split, label_value) -> str:
    """Devuelve el nombre legible de la etiqueta (si es ClassLabel)."""
    feat = dset_split.features.get("label")
    if isinstance(feat, ClassLabel):
        return feat.names[int(label_value)]
    return str(label_value)


def _sanitize(name: str) -> str:
    """Sanea nombres de archivo para Windows/Linux."""
    return (
        name.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )


def save_samples(
    out_dir: Path,
    n_train: int,
    n_test: int,
    hf_path: str = "GVJahnavi/PlantVillage_dataset",
) -> Tuple[int, Path]:
    """Guarda n_train + n_test imágenes en out_dir a partir del dataset.

    Args:
        out_dir: Carpeta destino donde se guardarán las imágenes.
        n_train: Nº de imágenes que se tomarán del split `train`.
        n_test: Nº de imágenes que se tomarán del split `test`.
        hf_path: Ruta del dataset en Hugging Face Hub.

    Returns:
        Un tuple con (total_guardadas, ruta_de_salida).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_dataset(hf_path)
    total = 0

    # Train
    if "train" in raw and n_train > 0:
        split = raw["train"]
        n = min(n_train, len(split))
        for i, ex in enumerate(split.select(range(n))):
            img = _as_pil(ex["image"])
            label_name = _sanitize(_get_label_name(split, ex["label"]))
            path = out_dir / f"train_{i:03d}_{label_name}.png"
            img.save(path)
            total += 1

    # Test
    if "test" in raw and n_test > 0:
        split = raw["test"]
        n = min(n_test, len(split))
        for i, ex in enumerate(split.select(range(n))):
            img = _as_pil(ex["image"])
            label_name = _sanitize(_get_label_name(split, ex["label"]))
            path = out_dir / f"test_{i:03d}_{label_name}.png"
            img.save(path)
            total += 1

    return total, out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Guardar muestras del dataset en data/test_images/"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/test_images",
        help="Carpeta de salida (default: data/test_images)",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=5,
        help="Número de imágenes a guardar del split train (default: 5)",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=5,
        help="Número de imágenes a guardar del split test (default: 5)",
    )
    parser.add_argument(
        "--hf-path",
        type=str,
        default="GVJahnavi/PlantVillage_dataset",
        help="Ruta del dataset en Hugging Face Hub",
    )
    args = parser.parse_args()

    total, out_dir = save_samples(
        out_dir=Path(args.out),
        n_train=args.n_train,
        n_test=args.n_test,
        hf_path=args.hf_path,
    )
    print(f"✓ Guardadas {total} imágenes en: {out_dir.resolve()}")


if __name__ == "__main__":
    main()