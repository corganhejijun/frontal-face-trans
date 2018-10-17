# -*- coding: utf-8 -*- 
from src.file_oper import copy_celeba
from src.util import faceFromDir

CELEBA_IDENTITY_FILE = "datasets\\celeba\\identity_CelebA.txt"
CELEBA_DATASET = "datasets\\celeba\\img_align_celeba"
MIDDLE_DIR = "datasets\\celeba\\celeba_identified"
OUT_DIR = "datasets\\celeba_aligned"

copy_celeba(CELEBA_IDENTITY_FILE, CELEBA_DATASET, MIDDLE_DIR)

OUT_DIR = "datasets\\celeba_aligned"
SHAPE_MODEL = "models\\shape_predictor_68_face_landmarks.dat"

faceFromDir(MIDDLE_DIR, OUT_DIR, SHAPE_MODEL)
