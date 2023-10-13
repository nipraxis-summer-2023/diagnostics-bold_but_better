""" Init for findoutlie module
"""
__version__ = '0.1a0'

from .outfind import find_outliers, load_fmri_data, get_data_files, load_event_data, hrf, convolved_time_course, find_outliers, remove_outliers, glm, evaluate_outlier_methods
from .detectors import iqr_detector, z_score_detector, dvars_detector
