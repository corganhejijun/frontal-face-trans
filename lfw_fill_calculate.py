# -*- coding: utf-8 -*- 
import os
from src.FaceDistance import getDatasetDistance

LFW_DATASET = "compare/result/lfw_fill_full"
MODEL_DIR = "models/20180402-114759"
RESULT_FILE_PATH = "compare/result/lfw_fill_distance_full.txt"

getDatasetDistance(RESULT_FILE_PATH, LFW_DATASET, MODEL_DIR)