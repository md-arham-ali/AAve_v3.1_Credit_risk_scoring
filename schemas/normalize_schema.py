"""Reference column structure for `normalize.ipynb`.

Identical to `process_schema` by construction, and re-exported rather than duplicated:
normalization rescales VALUES in place — bps/RAY/token-decimal division — and renames,
drops and adds nothing. A second copy of the same 16 tables would be a second thing to keep
in step, and the first time the two drifted the gate would be asserting against whichever
file the notebook happened to import.

That is also why `normalize.ipynb` needs no separate output check: its input check IS its
output check. The dtype tuples in `process_schema` exist precisely because this stage runs —
a rescaled column is `float64` where the raw one was `object`/`int64`, and both states are
structurally valid for the same file.
"""

from process_schema import SCHEMAS

__all__ = ["SCHEMAS"]
