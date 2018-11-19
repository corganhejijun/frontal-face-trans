# -*- coding: utf-8 -*- 
from src.FaceDistance import getDatasetDistance
from src.file_oper import copy_celeba
from src.util import faceFromDir

CELEBA_IDENTITY_FILE = "datasets/celeba/identity_CelebA.txt"
CELEBA_DATASET = "datasets/celeba/img_fill_celeba"
DATASET = "datasets/celeba/celeba_fill_identified"

copy_celeba(CELEBA_IDENTITY_FILE, CELEBA_DATASET, DATASET)

MODEL_DIR = "models/20180402-114759"
RESULT_FILE_PATH = "datasets/celeba_fill_distance.txt"

getDatasetDistance(RESULT_FILE_PATH, DATASET, MODEL_DIR)