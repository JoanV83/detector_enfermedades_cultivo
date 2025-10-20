"""Definición del modelo Vision Transformer (ViT) para clasificación.

Este módulo provee:
- ``VitConfig``: configuración mínima (checkpoint y número de clases).
- ``VitClassifier``: wrapper ligero sobre un backbone de Hugging Face que
  ajusta automáticamente la capa final al número de clases.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModelForImageClassification


@dataclass
class VitConfig:
    """Configuración mínima para construir el clasificador ViT.

    Attributes:
        model_name: Checkpoint (p. ej. "google/vit-base-patch16-224-in21k").
        num_labels: Número de clases para la capa final.
    """

    model_name: str
    num_labels: int


class VitClassifier(nn.Module):
    """Wrapper del backbone ViT para tareas de clasificación.

    La interfaz del ``forward`` es compatible con ``transformers``:
    recibe ``pixel_values`` (y opcionalmente ``labels``) y retorna un
    objeto con ``logits`` y, si se pasan etiquetas, ``loss``.
    """

    def __init__(self, config: VitConfig) -> None:
        """Inicializa el backbone con la capa final ajustada.

        Args:
            config: Instancia de ``VitConfig`` con checkpoint y clases.
        """
        super().__init__()
        self.model = AutoModelForImageClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
            ignore_mismatched_sizes=True,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        """Propagación hacia adelante.

        Args:
            pixel_values: Tensores de imagen (B, C, H, W) preprocesados.
            labels: Etiquetas enteras opcionales (para calcular ``loss``).

        Returns:
            ``ImageClassifierOutput`` de ``transformers`` con
            ``logits`` (y ``loss`` si se pasaron ``labels``).
        """
        return self.model(pixel_values=pixel_values, labels=labels)
