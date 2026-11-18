# Smart House: web panel, live sensors (HTTP/MQTT) and CLI on CPU PyTorch.
#   docker compose up            (see docker-compose.yml and docs/docker.md)
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# CPU-only PyTorch keeps the image small (no CUDA libraries).
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
COPY requirements.txt constraints.txt ./
RUN pip install torch --index-url "$TORCH_INDEX_URL" \
 && pip install -r requirements.txt paho-mqtt

COPY main.py ./
COPY smart_house ./smart_house
COPY data/models ./data/models
COPY examples ./examples

# Measurements, live readings, weather cache and new experiments persist here.
VOLUME ["/app/data/user"]
RUN useradd --create-home --uid 1000 house && mkdir -p /app/data/user && chown -R house /app/data/user
USER house

EXPOSE 8000
HEALTHCHECK --interval=60s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/live', timeout=4)"
ENTRYPOINT ["python", "main.py"]
CMD ["web", "--host", "0.0.0.0", "--port", "8000", "--weather", "open-meteo"]
