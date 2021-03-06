# -*- coding: utf-8 -*- 
from src.FaceDistance import getDatasetDistance

LFW_DATASET = "datasets/lfw"
MODEL_DIR = "models/20180402-114759"
RESULT_FILE_PATH = "compare/result/lfw_distance.txt"

getDatasetDistance(RESULT_FILE_PATH, LFW_DATASET, MODEL_DIR)