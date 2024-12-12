# src/feature_detection/__init__.py

from .sift_detection import detect_and_compute_sift, draw_keypoints as draw_sift_keypoints
from .surf_detection import detect_and_compute_surf, draw_keypoints as draw_surf_keypoints
from .orb_detection import detect_and_compute_orb, draw_keypoints as draw_orb_keypoints
from .siamese_matching import train_siamese_network, predict_similarity

__all__ = [
    'detect_and_compute_sift',
    'draw_sift_keypoints',
    'detect_and_compute_surf',
    'draw_surf_keypoints',
    'detect_and_compute_orb',
    'draw_orb_keypoints',
    'train_siamese_network',
    'predict_similarity'
]
