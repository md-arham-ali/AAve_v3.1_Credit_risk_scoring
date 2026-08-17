# credit-risk-model — second of the two domain-specific boxes, and the heavy one.
# Trains the full zoo, blends the PRI, reads the results back.
#
# Build (base image must exist first):
#   docker build -t aave-core:latest              -f docker/base/Dockerfile .
#   docker build -t aave-credit-risk-model:latest -f docker/credit_risk_model/Dockerfile .
#
# Expect 3-6 minutes on the first build: torch, xgboost, lightgbm, lifelines and
# scikit-learn are a genuinely large install. Rebuilds after a notebook or src/
# edit are instant — those are bind-mounted, not baked (see docker-compose.yml).

FROM aave-core:latest

# libgomp1 (the OpenMP runtime xgboost/lightgbm link against) already came from
# the base layer — the Linux equivalent of the `brew install libomp` step noted in
# requirements.txt. Nothing extra needed at the system level here.
COPY docker/credit_risk_model/requirements.txt /tmp/stage-requirements.txt
RUN pip install --no-cache-dir -r /tmp/stage-requirements.txt

# THE DOMAIN. model_config.py lives here rather than in src/, so the seven
# src/model_* modules resolve `from model_config import ...` from this path.
# Mirrors the repo layout so the notebooks' bootstrap cell works identically on
# the host and in the container. Point this at domains/<other>/ and the same
# image definition serves a different domain — no src/ changes.
COPY domains/credit_risk/ /app/domains/credit_risk/

ENV PYTHONPATH=/app/src:/app/domains/credit_risk

COPY notebooks/model_training.ipynb /app/notebooks/model_training.ipynb
COPY notebooks/model_results.ipynb  /app/notebooks/model_results.ipynb

COPY docker/credit_risk_model/run.sh /app/run.sh
RUN chmod +x /app/run.sh

CMD ["/app/run.sh"]
