# Base mínima
FROM python:3.10-slim

# ---- Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# ---- Directorio de trabajo
WORKDIR /app

# ---- Dependencias del sistema (mínimas)
# Si usas OpenCV, deja libgl1; si NO lo usas, puedes quitarlo.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl libgl1 \
  && rm -rf /var/lib/apt/lists/*

# ---- Copiamos dependencias primero para aprovechar caché
COPY requirements.txt /app/requirements.txt

# ---- Instala dependencias Python (CPU-only)
# Asegúrate de que requirements.txt ya tenga torch cpu:
# torch==2.3.1+cpu, torchvision==0.18.1+cpu, torchaudio==2.3.1+cpu
# -f https://download.pytorch.org/whl/torch_stable.html
RUN python -m pip install --upgrade pip \
 && pip install -r requirements.txt

# ---- Copia el proyecto
COPY . /app

# ---- Instala el paquete en editable (si lo necesitas como módulo)
RUN pip install -e .

# ---- Crea directorios de trabajo
RUN mkdir -p /app/checkpoints /app/runs /app/artifacts /app/data

# ---- Usuario no-root por seguridad
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# ---- Exponer puerto y healthcheck
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ---- Comando por defecto
CMD ["python", "-m", "streamlit", "run", "src/plant_disease/apps/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]