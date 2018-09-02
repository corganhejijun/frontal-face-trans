# -*- coding: utf-8 -*- 
import os
from src.util import getDatasetDistance

DATASET = "datasets\\celeba_aligned"
MODEL_DIR = "models\\20180402-114759"
RESULT_FILE_PATH = "datasets\\celeba_distance.txt"

getDatasetDistance(RESULT_FILE_PATH, DATASET, MODEL_DIR)