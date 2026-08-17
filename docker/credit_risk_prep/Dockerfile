# credit-risk-prep — first of the two domain-specific boxes.
# Shared panel -> split, leakage-guarded modelling table for the credit-risk domain.
#
# Build (base image must exist first):
#   docker build -t aave-core:latest             -f docker/base/Dockerfile .
#   docker build -t aave-credit-risk-prep:latest -f docker/credit_risk_prep/Dockerfile .

FROM aave-core:latest

COPY docker/credit_risk_prep/requirements.txt /tmp/stage-requirements.txt
RUN pip install --no-cache-dir -r /tmp/stage-requirements.txt

# THE DOMAIN. model_config.py lives here, not in src/ — the seven src/model_*
# modules do `from model_config import ...` and resolve it from this path. Copy
# to a path mirroring the repo layout so the notebooks' bootstrap cell
# (sys.path.insert(_ROOT / 'domains' / 'credit_risk')) works identically inside
# the container and on the host.
COPY domains/credit_risk/ /app/domains/credit_risk/

# Extend the base image's PYTHONPATH rather than replacing it: src/ stays
# importable, and the domain config joins it. Swapping this one line to another
# domains/<name>/ is what makes a second domain a new image, not a code fork.
ENV PYTHONPATH=/app/src:/app/domains/credit_risk

COPY notebooks/model_features.ipynb /app/notebooks/model_features.ipynb
COPY notebooks/model_dataset.ipynb  /app/notebooks/model_dataset.ipynb
COPY notebooks/model_split.ipynb    /app/notebooks/model_split.ipynb

COPY docker/credit_risk_prep/run.sh /app/run.sh
RUN chmod +x /app/run.sh

CMD ["/app/run.sh"]
