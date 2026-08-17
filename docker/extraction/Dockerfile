# Extraction stage: Dune API fetch + raw-unit normalization.
#
# Build (base image must exist first):
#   docker build -t aave-core:latest      -f docker/base/Dockerfile .
#   docker build -t aave-extraction:latest -f docker/extraction/Dockerfile .

FROM aave-core:latest

# Stage-only deps. pandas/numpy/papermill already came from the base layer, so
# this installs just two small packages — seconds, not minutes.
COPY docker/extraction/requirements.txt /tmp/stage-requirements.txt
RUN pip install --no-cache-dir -r /tmp/stage-requirements.txt

# The notebooks that DEFINE this stage. Baked in so the image runs standalone;
# docker-compose bind-mounts ./notebooks over this at run time so edits are live
# (same dual pattern as src/).
COPY notebooks/process.ipynb   /app/notebooks/process.ipynb
COPY notebooks/normalize.ipynb /app/notebooks/normalize.ipynb

# chmod here rather than relying on the host file's permission bit, so the image
# builds correctly regardless of how run.sh is checked out.
COPY docker/extraction/run.sh /app/run.sh
RUN chmod +x /app/run.sh

CMD ["/app/run.sh"]
