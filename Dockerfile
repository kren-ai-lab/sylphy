# Runtime image of a released Sylphy version, installed from PyPI (no source copied), so any past
# release can be rebuilt. Built and pushed by .github/workflows/publish-container.yml.
#   docker build --build-arg SYLPHY_VERSION=1.2.0 -t sylphy:1.2.0 .
#   docker build --build-arg SYLPHY_VERSION=1.2.0 --build-arg EXTRAS=[all] -t sylphy:1.2.0-full .
FROM python:3.14-slim

ARG SYLPHY_VERSION
ARG EXTRAS=""
LABEL org.opencontainers.image.source="https://github.com/kren-ai-lab/sylphy" \
      org.opencontainers.image.description="Sylphy: protein sequence representation (encoders, embeddings, reductions)" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${SYLPHY_VERSION}"

# procps: Nextflow reads task metrics with `ps` inside the container.
RUN apt-get update \
    && apt-get install -y --no-install-recommends procps \
    && rm -rf /var/lib/apt/lists/*

# gcc: the `all`/`reductions` extras pull sdist-only packages (clustpy, biotraj) that compile C; removed afterwards.
RUN test -n "${SYLPHY_VERSION}" \
    && if [ -n "${EXTRAS}" ]; then apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev; fi \
    && pip install --no-cache-dir "sylphy${EXTRAS}==${SYLPHY_VERSION}" \
    && if [ -n "${EXTRAS}" ]; then apt-get purge -y --auto-remove gcc libc6-dev; fi \
    && rm -rf /var/lib/apt/lists/*
