# -*- coding: utf-8 -*- 
import os
from src.FaceDistance import getDatasetDistance

LFW_DATASET = "compare/result/LFW_FF_GAN"
MODEL_DIR = "models/20180402-114759"
RESULT_FILE_PATH = "compare/result/lfw_compare_FF_Gan_distance.txt"

getDatasetDistance(RESULT_FILE_PATH, LFW_DATASET, MODEL_DIR)