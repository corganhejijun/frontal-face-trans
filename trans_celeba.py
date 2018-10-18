# -*- coding: utf-8 -*-
from src.util import transFromDir

DATASET = "datasets/celeba/celeba_identified"
ALIGNED_DATA = "datasets/celeba_aligned"
OUT_DIR = "datasets/celeba_trans"
SHAPE_MODEL = "models/shape_predictor_68_face_landmarks.dat"

EIGEN_PATH = "datasets/eigen_face.jpg"

transFromDir(DATASET, ALIGNED_DATA, OUT_DIR, SHAPE_MODEL, EIGEN_PATH)