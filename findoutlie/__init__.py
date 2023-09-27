""" Init for findoutlie module
"""
__version__ = '0.1a0'

from .outfind import find_outliers
from .detect_outliers import detect_outliers
from .metrics import dvars
from .detectors import iqr_detector