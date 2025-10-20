"""Pruebas de inferencia con el modelo entrenado.

Este test verifica que la función `predict` devuelve un top-K válido cuando
se le pasa una imagen dummy. Se omite si no existe el directorio del modelo.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from plant_disease.inference.predict import (
    DEFAULT_MODEL_DIR,
    predict,
)

MODEL_DIR = DEFAULT_MODEL_DIR


@pytest.mark.skipif(
    not os.path.exists(MODEL_DIR),
    reason="Modelo no encontrado en runs/vit-gvj/final",
)
def test_predict_dummy() -> None:
    """Debe devolver una lista top-K con (clase, probabilidad)."""
    # Imagen dummy 224x224 gris
    img = Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8))

    # En Windows, PIL puede fallar si el archivo está abierto por otro handle.
    # Por eso se usa TemporaryDirectory y se guarda con `save` a una ruta.
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "dummy.png")
        img.save(img_path)

        preds: list[tuple[str, float]] = predict(img_path, model_dir=MODEL_DIR, topk=3)

        assert isinstance(preds, list)
        assert len(preds) == 3

        # Cada item debe ser (label: str, prob: float)
        label, prob = preds[0]
        assert isinstance(label, str)
        assert isinstance(prob, float)
