"""Pruebas de utilidades de ``inference.predict`` (carga de clases).

Se validan dos comportamientos clave sin requerir un modelo real:

1) Preferencia por ``labels.json`` en el directorio del modelo:
   si existe, debe usarse para obtener los nombres de clase.

2) Fallback a la lectura desde el dataset del Hub (mockeado):
   si no hay ``labels.json``, se obtienen los nombres desde
   ``ClassLabel.names`` del split de entrenamiento.

Las pruebas mockean ``load_dataset`` para retornar un ``DatasetDict`` en
memoria con la columna ``label`` como ``ClassLabel``, evitando red/IO.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image as HFImage

from plant_disease.inference.predict import (
    _load_class_names,
    _load_class_names_from_hub,
)


def _hub_like_dataset_with_classlabel() -> DatasetDict:
    """Devuelve un DatasetDict con ``ClassLabel(names=['x','y'])``.

    Se crea un mini-split train con una imagen dummy (matriz 8x8) y etiqueta
    ``0``. El objetivo es que ``_load_class_names_from_hub`` pueda leer
    ``ClassLabel.names`` y devolver ``["x", "y"]``.
    """
    features = Features({"image": HFImage(), "label": ClassLabel(names=["x", "y"])})
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    d = Dataset.from_dict({"image": [img], "label": [0]}, features=features)
    return DatasetDict({"train": d})


def test_load_class_names_prefers_labels_json(tmp_path):
    """Usa ``labels.json`` si está presente en el directorio del modelo."""
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True)
    labels = ["A", "B", "C"]
    (model_dir / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False), encoding="utf-8"
    )

    got = _load_class_names(str(model_dir))
    assert got == labels


def test_load_class_names_from_hub_with_classlabel(monkeypatch):
    """Hace fallback a ``ClassLabel`` del dataset cuando no hay labels.json."""
    def _fake_load_dataset(_: str, **__: Any) -> DatasetDict:
        return _hub_like_dataset_with_classlabel()

    monkeypatch.setattr(
        "plant_disease.inference.predict.load_dataset", _fake_load_dataset
    )

    got = _load_class_names_from_hub("FAKE/DS")
    assert got == ["x", "y"]
