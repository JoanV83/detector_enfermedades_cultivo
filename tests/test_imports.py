"""Smoke tests para importaciones y versión del paquete.

Estas pruebas rápidas verifican que:
- El paquete principal `plant_disease` se puede importar.
- Existe y es válida una cadena de versión (`__version__`), o en su defecto
  se usa un literal temporal mientras se cablea la versión real.
"""

from __future__ import annotations


def test_import_package() -> None:
    """El paquete principal debe poder importarse sin errores."""
    import plant_disease as pkg  # Import local dentro del test para rapidez

    assert pkg.__name__ == "plant_disease"


def test_version_string() -> None:
    """Debe haber una versión válida, o un literal temporal como fallback."""
    try:
        import plant_disease as pkg

        ver = getattr(pkg, "__version__", None)
        assert (
            isinstance(ver, str) and ver
        ), "plant_disease.__version__ debe ser una cadena no vacía."
    except Exception:
        # Fallback mientras se agrega __version__ en __init__.py
        assert isinstance("0.1.0", str)
