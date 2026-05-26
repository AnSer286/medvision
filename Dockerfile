# Многоступенчатая сборка для уменьшения итогового образа
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

FROM python:3.12-slim
# Безопасность: непривилегированный пользователь (CIS Docker Benchmark 4.1)
RUN useradd --create-home --uid 1000 appuser
WORKDIR /srv
COPY --from=builder /install /usr/local
COPY app/ app/
COPY models/ models/
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
