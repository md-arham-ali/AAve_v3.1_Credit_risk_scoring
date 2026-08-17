# MLflow tracking server — the one LONG-RUNNING box.
#
# The five pipeline stages are run-to-completion (`docker compose run --rm X`,
# exit 0 = done). This one stays up serving the UI and the tracking API:
#   docker compose up -d mlflow      # start, detached
#   docker compose down              # stop
#
# Deliberately NOT built FROM aave-core: the server never imports src/, never
# touches the domain config, and never needs pandas. Keeping it on plain python
# means an MLflow upgrade can't disturb the pipeline images, and vice versa.
#
# Build:
#   docker build -t aave-mlflow:latest -f docker/mlflow/Dockerfile .

FROM python:3.12-slim

# mlflow[extras] is not needed — the plain server plus the SQLite backend covers
# local file-store tracking. Pinned so the UI and the client in
# credit-risk-model stay on the same schema version.
RUN pip install --no-cache-dir mlflow==2.22.0

# /mlflow is bind-mounted to ./mlflow_data on the host, so both the run database
# and every logged artifact are plain files on your disk (the local-storage
# requirement). Nothing lives inside the container.
WORKDIR /mlflow

EXPOSE 5001

# --backend-store-uri   run/param/metric metadata -> SQLite file
# --default-artifact-root  models, scorecards, prediction CSVs -> a directory
# --host 0.0.0.0        bind to all interfaces INSIDE the container, otherwise
#                       the published port on the host reaches nothing
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--default-artifact-root", "/mlflow/artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5001"]
