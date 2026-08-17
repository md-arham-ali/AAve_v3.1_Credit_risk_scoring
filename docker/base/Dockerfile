# Shared base image: src/ + the deps every pipeline stage needs.
# Every stage Dockerfile starts `FROM aave-core:latest` and adds only its own deps.
#
# Build from the REPO ROOT (the build context must include src/):
#   docker build -t aave-core:latest -f docker/base/Dockerfile .

FROM python:3.12-slim

# libgomp1 = the OpenMP runtime xgboost/lightgbm link against (the Linux
# equivalent of `brew install libomp` noted in requirements.txt). Lives in base
# so it's one cached layer, not repeated per stage.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- dependency layer FIRST ---------------------------------------------------
# Requirements change rarely; code changes constantly. Copying requirements
# before src/ means editing a .py file reuses this cached pip layer instead of
# reinstalling pandas every build.
COPY docker/base/requirements.txt /tmp/base-requirements.txt
RUN pip install --no-cache-dir -r /tmp/base-requirements.txt

# --- code layer LAST ----------------------------------------------------------
# Baked in so the image works standalone with no repo checkout. At run time,
# docker-compose bind-mounts the host's ./src over this path — because
# PYTHONPATH resolves imports by path on every import, the mount transparently
# shadows this copy and your live edits take effect with no rebuild.
COPY src/ /app/src/

# PYTHONPATH           — makes src/ importable flat (import model_config), matching
#                        how the modules already import each other
# PYTHONUNBUFFERED     — your notebooks' print() progress appears live in container
#                        logs instead of buffering until the process exits
# PYTHONDONTWRITEBYTECODE — stops containers littering __pycache__/ into the
#                        bind-mounted (i.e. your real) src/ directory
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Default command is a self-test: proves src/ is importable in this image.
CMD ["python", "-c", "import model_config, transform, EDA; print('aave-core OK — src/ importable, pandas', __import__('pandas').__version__)"]
