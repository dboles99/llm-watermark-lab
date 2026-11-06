FROM python:3.12-slim@sha256:dd29372629eeba2dd003fd9e9d35a5b8236c44727875a0364254b5127af88e65

LABEL org.opencontainers.image.title="Local non-hardware TEE simulator" \
      org.opencontainers.image.description="Local contract test server; never hardware attestation" \
      org.watermark-lab.attestation-class="simulated-non-hardware" \
      org.watermark-lab.promotion-eligible="false"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    WATERMARK_LAB_TEE_DEPLOYMENT_MODE=local-development

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app \
    && python -m pip install --no-cache-dir \
       "fastapi==0.141.1" \
       "pydantic==2.13.4" \
       "uvicorn==0.52.3"

COPY --chown=app:app src/watermark_lab /app/src/watermark_lab
COPY --chown=app:app config/calibration/kgw.toml /app/config/calibration/kgw.toml

USER app
EXPOSE 8787

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8787/health', timeout=2).read()"]

CMD ["python", "-m", "uvicorn", "watermark_lab.tee_server.app:app", "--host", "0.0.0.0", "--port", "8787", "--workers", "1", "--no-access-log", "--no-server-header"]
