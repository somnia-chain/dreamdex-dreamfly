# DreamFly worker: the fly, its order mirror and the dashboard in one container (see docs/railway.md).
# Node is copied in only for the perp and event contract sidecars (recorders, perp order executor); a spot-only
# run never starts it.

FROM node:24-bookworm-slim AS recorders
WORKDIR /app/perp
COPY perp/package.json perp/package-lock.json ./
RUN npm ci --omit=dev

FROM python:3.13-slim-bookworm

COPY --from=recorders /usr/local/bin/node /usr/local/bin/node

WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE.md ./
COPY flybot flybot
# editable: the code finds data/, config/ and site/ relative to /app, so the package must stay in /app
RUN pip install --no-cache-dir -e ".[live]"

COPY scripts scripts
COPY site site
COPY config config
COPY data/mb data/mb
COPY perp perp
COPY --from=recorders /app/perp/node_modules perp/node_modules

# Point STATE_DIR at a mounted volume to keep what the fly learns across deploys.
ENV PYTHONUNBUFFERED=1 \
    NETWORK=testnet \
    DRY_RUN=true \
    PORT=8765
EXPOSE 8765

CMD ["dreamfly", "start"]
