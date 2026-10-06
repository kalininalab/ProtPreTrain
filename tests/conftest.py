"""Pytest setup: the smoke tests are CPU-only."""

import os

# Hide GPUs before torch is imported: a visible GPU without kernels in this torch build (e.g. sm_61) still reports
# torch.cuda.is_available() and would be picked by code that moves models to "cuda" when available.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
