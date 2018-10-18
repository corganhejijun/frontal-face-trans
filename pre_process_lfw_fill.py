# -*- coding: utf-8 -*- 
from src.resize_for_train import readMarginFile

RESIZE_MARGIN = "datasets/lfw_resize_margin.txt"
ORIGIN_DATA = "datasets/lfw_fill"
OUT_FOLDER = "datasets/lfw_fill_aligned"
LFW_DATASET = "datasets/lfw_aligned"

        
readMarginFile(RESIZE_MARGIN, ORIGIN_DATA, OUT_FOLDER, LFW_DATASET)
