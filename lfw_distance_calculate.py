# -*- coding: utf-8 -*- 
from src.FaceDistance import distance

LFW_DATASET = "datasets\\lfw\\Abdullah_Gul"
MODEL_DIR = "models\\20180402-114759"

avg, std = distance(LFW_DATASET, MODEL_DIR)
print("average = %f" % avg)
print("standard = %f" % std)