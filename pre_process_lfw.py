# -*- coding: utf-8 -*- 
from src.util import faceFromDir

IN_DIR = "datasets\\lfw"
OUT_DIR = "datasets\\lfw_aligned"
SHAPE_MODEL = "models\\shape_predictor_68_face_landmarks.dat"

faceFromDir(IN_DIR, OUT_DIR, SHAPE_MODEL)
            