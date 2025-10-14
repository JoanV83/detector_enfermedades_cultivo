"""Pruebas del módulo ``plant_disease.data.datasets``.

Este archivo valida el flujo esencial del cargador genérico:

- Normaliza columnas a ``image`` (PIL) y ``label`` (int o str).
- Genera un split de validación si no existe, usando el 10% del train.
- Construye ``class_names`` para tres variantes de etiqueta:
  1) Enteros (se crean nombres ``class_i``).
  2) Cadenas de texto (vocabulario unificado → IDs).
  3) ``ClassLabel`` de Hugging Face (usa sus ``names``).

Para evitar Internet, se hace monkeypatch a ``datasets.load_dataset`` y se
construye un ``DatasetDict`` mínimo en memoria.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
from PIL import Image
from datasets import ClassLabel, Dataset, DatasetDict, Features
from datasets import Image as HFImage
from datasets import Value

from plant_disease.data.datasets import load_generic_from_hub


# ---------------------------------------------------------------------------
# Helpers para construir splits artificiales
# ---------------------------------------------------------------------------

def _tiny_split_with_int_labels() -> Dataset:
    """Devuelve un split con imágenes PIL y etiquetas enteras 0/1.

    Las imágenes son matrices 8x8 negras convertidas a PIL.
    Las etiquetas se almacenan como ``Value("int64")`` para simular el caso
    de enteros sin ``ClassLabel``.
    """
    imgs = [Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)) for _ in range(10)]
    labels = [0, 1, 1, 0, 1, 0, 1, 0, 1, 0]
    features = Features({"image": HFImage(), "label": Value("int64")})
    return Dataset.from_dict({"image": imgs, "label": labels}, features=features)


def _tiny_split_with_text_labels() -> Dataset:
    """Devuelve un split con imágenes PIL y etiquetas de texto ``'a'``/``'b'``.

    Representa datasets con ``label`` tipo string, donde el cargador debe
    construir un vocabulario global y remapear a IDs enteros consistentes.
    """
    imgs = [Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)) for _ in range(10)]
    labels = ["a", "b", "b", "a", "b", "a", "b", "a", "b", "a"]
    features = Features({"image": HFImage(), "label": Value("string")})
    return Dataset.from_dict({"image": imgs, "label": labels}, features=features)


def _tiny_split_with_classlabel() -> Dataset:
    """Devuelve un split con ``ClassLabel(names=['a', 'b'])``.

    En este escenario, se espera que el loader detecte el ``ClassLabel`` y use
    sus ``names`` como ``class_names`` sin remapeos adicionales.
    """
    imgs = [Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)) for _ in range(10)]
    labels = [0, 1, 1, 0, 1, 0, 1, 0, 1, 0]
    features = Features({"image": HFImage(), "label": ClassLabel(names=["a", "b"])})
    return Dataset.from_dict({"image": imgs, "label": labels}, features=features)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "builder, expected_names",
    [
        (_tiny_split_with_int_labels, ["class_0", "class_1"]),
        (_tiny_split_with_text_labels, sorted(["a", "b"])),
        (_tiny_split_with_classlabel, ["a", "b"]),
    ],
)
def test_load_generic_from_hub_variants(
    monkeypatch: pytest.MonkeyPatch,
    builder: Callable[[], Dataset],
    expected_names: list[str],
) -> None:
    """Valida normalización de columnas y resolución de ``class_names``.

    Se sustituye ``load_dataset`` por una función que devuelve sólo un split
    ``train``. El loader debe:
      - renombrar a ``image``/``label`` cuando proceda;
      - crear ``val`` a partir de ``train`` (10%);
      - producir ``class_names`` correctos según el tipo de etiqueta.
    """

    def _fake_load_dataset(_: str, cache_dir=None) -> DatasetDict:  # noqa: ARG001
        split = builder()
        # Sólo train; el loader debe crear val a partir de train.
        return DatasetDict({"train": split})

    monkeypatch.setattr(
        "plant_disease.data.datasets.load_dataset",
        _fake_load_dataset,
        raising=True,
    )

    # Ejecutar loader
    splits, class_names = load_generic_from_hub("FAKE/PATH")

    # Debe existir train y val (val generado desde 10% del train).
    assert "train" in splits and "val" in splits
    assert set(class_names) == set(expected_names)

    # Columnas limpias en todos los splits.
    for name, dset in splits.items():
        assert set(dset.column_names) == {"image", "label"}, name

    # Tamaños coherentes con partición 90/10 (redondeo hacia abajo + mínimo 1).
    n_total = 10
    n_val_expected = max(1, int(0.1 * n_total))
    assert len(splits["val"]) == n_val_expected
    assert len(splits["train"]) == n_total - n_val_expected