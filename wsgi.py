"""
wsgi.py — Gunicorn entry point for BIMLearn
============================================

Production launch:
    gunicorn \
      --workers 1 \
      --threads 4 \
      --timeout 120 \
      --bind 0.0.0.0:8000 \
      --access-logfile - \
      --error-logfile  - \
      wsgi:app

Environment variables:
    BIM_MODEL_PATH  path to .keras model file   (default: efficientnet_b0_bim_static_model_finetuned.keras)
    DATABASE_URL    SQLAlchemy DB URL            (default: sqlite:///bimlearn.db)
    SECRET_KEY      Flask session secret         (CHANGE THIS in production)

Worker count note:
    Keep workers=1 so TensorFlow loads once.
    Use threads=4 for concurrent request handling.
    Scale horizontally (multiple machines) rather than vertically (multiple workers)
    if you need higher throughput — each machine loads its own TF session.
"""

from app import app  # noqa: F401
