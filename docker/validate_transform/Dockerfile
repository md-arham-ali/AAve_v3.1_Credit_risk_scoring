# Validation + transformation stage: raw tables -> validated, feature-complete panels.
#
# Merged into one box because the two concerns aren't separable in the code —
# transform.ipynb already calls adv_validation's tier checks for its own
# self-verification, and validation.ipynb validates the frames transform builds.
#
# Build (base image must exist first):
#   docker build -t aave-core:latest              -f docker/base/Dockerfile .
#   docker build -t aave-validate-transform:latest -f docker/validate_transform/Dockerfile .

FROM aave-core:latest

COPY docker/validate_transform/requirements.txt /tmp/stage-requirements.txt
RUN pip install --no-cache-dir -r /tmp/stage-requirements.txt

# The three notebooks that define this stage, in execution order.
COPY notebooks/transform.ipynb         /app/notebooks/transform.ipynb
COPY notebooks/feature_addition.ipynb  /app/notebooks/feature_addition.ipynb
COPY notebooks/validation.ipynb        /app/notebooks/validation.ipynb

COPY docker/validate_transform/run.sh /app/run.sh
RUN chmod +x /app/run.sh

CMD ["/app/run.sh"]
