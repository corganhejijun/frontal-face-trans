# -*- coding: utf-8 -*- 
import os
from src.util import getDatasetDistance

LFW_DATASET = "datasets\\lfw_fill_aligned"
MODEL_DIR = "models\\20180402-114759"
RESULT_FILE_PATH = "datasets\\lfw_fill_distance.txt"

getDatasetDistance(RESULT_FILE_PATH, LFW_DATASET, MODEL_DIR)