# -*- coding: utf-8 -*-
from src.util import transFromDir

DATASET = "datasets\\lfw"
ALIGNED_DATA = "datasets\\lfw_aligned"
OUT_DIR = "datasets\\lfw_trans"
SHAPE_MODEL = "models\\shape_predictor_68_face_landmarks.dat"

EIGEN_PATH = "datasets\\eigen_face.jpg"
OUT_SIZE = 256

transFromDir(DATASET, ALIGNED_DATA, OUT_DIR, SHAPE_MODEL, EIGEN_PATH, OUT_SIZE)