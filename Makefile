# ===== Makefile para proyecto Python con uv =====
# Uso local:    make test        # corre tests
#               make lint        # ruff + black --check
#               make format      # formatea (ruff fix + black)
#               make typecheck   # mypy (si lo tienes en deps)
#               make build       # wheel + sdist
# Objetivo por defecto:
.DEFAULT_GOAL := test

# Puedes ajustar estas variables si quieres
PYTEST_ARGS ?= -q --maxfail=1 --disable-warnings

.PHONY: setup test lint format typecheck build clean help

help:
	@echo "Targets:"
	@echo "  setup      -> uv sync (usa uv.lock si existe)"
	@echo "  test       -> pytest (usa uv run)"
	@echo "  lint       -> ruff + black --check"
	@echo "  format     -> ruff fix + black"
	@echo "  typecheck  -> mypy (opcional)"
	@echo "  build      -> uv build (wheel + sdist)"
	@echo "  clean      -> limpia caches"

setup:
	uv --version
	uv venv --allow-existing
	uv pip install -e ".[dev]"

test: setup
	.venv/bin/pytest $(PYTEST_ARGS)

lint:
	uvx ruff check src tests
	uvx black --check src tests

format:
	# Intenta autocorregir con ruff; si algo falla no detengas
	uvx ruff fix src tests || true
	uvx black src tests

typecheck:
	uvx mypy src || true

build: setup
	uv build

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist build
