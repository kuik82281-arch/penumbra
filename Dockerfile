FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.en.md LICENSE ./
COPY penumbra ./penumbra
RUN pip install --no-cache-dir .

# Memory lives in /data (mount a volume); the service listens on all interfaces inside the container.
ENV PENUMBRA_DATA=/data \
    PENUMBRA_HOST=0.0.0.0 \
    PENUMBRA_PORT=8790 \
    PENUMBRA_EMBEDDING=none
VOLUME ["/data"]
EXPOSE 8790

CMD ["python", "-m", "penumbra", "serve"]
