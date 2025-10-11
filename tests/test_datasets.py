"""Pruebas del módulo ``plant_disease.data.datasets``.

Este archivo valida el flujo mínimo del cargador genérico de datasets:

- Normaliza columnas a ``image`` (PIL) y ``label`` (int o str).
- Genera un split de validación si no existe usando el 10% del train.
- Construye ``class_names`` apropiadamente para tres variantes:
  * Etiquetas como enteros (se crean nombres ``class_i``).
  * Etiquetas como cadenas (mapeo global por vocabulario unificado).
  * Etiquetas con ``ClassLabel`` de Hugging Face (usa sus nombres).

Para evitar dependencia de Internet, las pruebas **mockean**
``datasets.load_dataset`` y construyen un ``DatasetDict`` mínimo en memoria.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pytest
from PIL import Image
from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    Features,
    Image as HFImage,
    Value,
)

from plant_disease.data.datasets import load_generic_from_hub


def _tiny_split_with_int_labels() -> Dataset:
    """Devuelve un split con imágenes PIL y etiquetas enteras 0/1.

    La función genera un conjunto toy con cuatro ejemplos. Las imágenes son
    matrices 8x8 en negro convertidas a PIL. Las etiquetas provienen de un
    ``Value("int64")`` para simular el caso de enteros sin ``ClassLabel``.
    """
    imgs = [Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)) for _ in range(4)]
    labels = [0, 1, 1, 0]
    features = Features({"image": HFImage(), "label": Value("int64")})
    return Dataset.from_dict({"image": imgs, "label": labels}, features=features)


def _tiny_split_with_text_labels() -> Dataset:
    """Devuelve un split con imágenes PIL y etiquetas de texto 'a'/'b'.

    Este caso representa datasets con ``label`` en formato string, donde el
    cargador debe construir un vocabulario global y remapear a IDs enteros.
    """
    imgs = [Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)) for _ in range(4)]
    labels = ["a", "b", "b", "a"]
    features = Features({"image": HFImage(), "label": Value("string")})
    return Dataset.from_dict({"image": imgs, "label": labels}, features=features)


def _tiny_split_with_classlabel() -> Dataset:
    """Devuelve un split con ``ClassLabel(names=['a', 'b'])``.

    En este escenario, esperamos que el loader detecte el ``ClassLabel`` y
    utilice sus ``names`` como ``class_names`` finales sin remapeos extra.
    """
    imgs = [Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)) for _ in range(4)]
    labels = [0, 1, 1, 0]
    features = Features({"image": HFImage(), "label": ClassLabel(names=["a", "b"])})
    return Dataset.from_dict({"image": imgs, "label": labels}, features=features)


@pytest.mark.parametrize(
    "builder, expected_names",
    [
        (_tiny_split_with_int_labels, ["class_0", "class_1"]),
        (_tiny_split_with_text_labels, sorted(["a", "b"])),
        (_tiny_split_with_classlabel, ["a", "b"]),
    ],
)
def test_load_generic_from_hub_variants(monkeypatch, builder, expected_names):
    """Valida normalización de columnas y resolución de ``class_names``.

    Se reemplaza ``load_dataset`` por una versión falsa que devuelve un único
    split ``train``. El loader debe:

    - Renombrar columnas a ``image`` y ``label``.
    - Crear ``val`` a partir de ``train`` (10%).
    - Producir ``class_names`` consistentes con cada variante de etiqueta.
    """
    def _fake_load_dataset(_: str, cache_dir=None) -> DatasetDict:
        split = builder()
        # Sólo train; el loader debe crear val a partir de train.
        return DatasetDict({"train": split})

    monkeypatch.setattr(
        "plant_disease.data.datasets.load_dataset", _fake_load_dataset
    )

    splits, class_names = load_generic_from_hub("FAKE/PATH")

    # Debe existir train y val (val generado desde train).
    assert "train" in splits and "val" in splits
    assert set(class_names) == set(expected_names)

    # Columnas limpias en todos los splits.
    for name, dset in splits.items():
        assert set(dset.column_names) == {"image", "label"}, name