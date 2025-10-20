"""Pruebas del wrapper ``VitClassifier`` del módulo de modelos.

Objetivo
--------
Asegurar que el wrapper inicializa un modelo compatible con la API de
Transformers y que el método ``forward`` acepta ``pixel_values`` y
devuelve un objeto con el atributo ``logits`` de la forma esperada.

Estrategia
----------
- Se mockea ``AutoModelForImageClassification.from_pretrained`` para
  devolver un modelo PyTorch mínimo (``_DummyHFModel``) que:
  * Aplana el tensor de entrada y aplica una capa lineal.
  * Devuelve un objeto simple con ``logits`` y ``loss=None``.
- Se crea un ``VitClassifier`` con 3 clases y se realiza un forward con un
  batch de 2 imágenes falsas (3x224x224), verificando la forma de logits.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from plant_disease.models.vit import VitClassifier, VitConfig


class _DummyHFModel(nn.Module):
    """Modelo minimalista con la misma firma que HF para pruebas.

    No realiza atención ni particionado por patches; únicamente aplana el
    tensor y aplica una proyección lineal a ``num_labels``. Este dummy
    permite probar el wrapper sin descargar pesos ni requerir Internet.
    """

    def __init__(self, num_labels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(3 * 224 * 224, num_labels)

    def forward(self, pixel_values: torch.Tensor, labels: torch.Tensor | None = None):
        """Simula el forward de un clasificador de imágenes.

        Parameters
        ----------
        pixel_values:
            Tensor de forma (B, 3, 224, 224).

        labels:
            Ignorado en este dummy. Se acepta para mantener compatibilidad
            con la firma típica de Transformers.

        Returns
        -------
        obj:
            Objeto simple con los atributos:
            - ``logits``: tensor (B, num_labels)
            - ``loss``: None (sin cálculo de pérdida en el dummy)
        """
        b = pixel_values.shape[0]
        x = pixel_values.view(b, -1)
        logits = self.linear(x)
        return type("Out", (), {"logits": logits, "loss": None})


def test_vit_classifier_forward(monkeypatch):
    """El wrapper debe crear modelo y propagar ``pixel_values`` correctamente."""

    def _fake_from_pretrained(model_name: str, num_labels: int, **_: Any):
        return _DummyHFModel(num_labels)

    monkeypatch.setattr(
        "plant_disease.models.vit.AutoModelForImageClassification.from_pretrained",
        _fake_from_pretrained,
    )

    clf = VitClassifier(VitConfig("any/model", num_labels=3))
    x = torch.randn(2, 3, 224, 224)
    out = clf(pixel_values=x)
    assert hasattr(out, "logits")
    assert out.logits.shape == (2, 3)
