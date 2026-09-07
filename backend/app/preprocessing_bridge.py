"""Re-export the preprocessing helpers the backend shares with training.

Training and serving MUST normalise identically. Importing the same function rather
than reimplementing it is what guarantees that; a second copy of the constants is a
silent-drift risk.
"""
from ml.preprocessing.face_processor import MEAN, STD, normalize  # noqa: F401
