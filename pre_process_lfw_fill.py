# -*- coding: utf-8 -*- 
from src.resize_for_train import readMarginFile
import os
from shutil import copyfile

RESIZE_MARGIN = "datasets/lfw_resize_margin.txt"
ORIGIN_DATA = "datasets/lfw_fill"
OUT_FOLDER = "datasets/lfw_fill_aligned"
LFW_DATASET = "datasets/lfw_aligned"

        
# readMarginFile(RESIZE_MARGIN, ORIGIN_DATA, OUT_FOLDER, LFW_DATASET)

ORIGIN_DATA = "datasets/lfw_fill_hd"
OUT_FOLDER = "datasets/lfw_fill_sub_hd"

if not os.path.exists(OUT_FOLDER):
    os.mkdir(OUT_FOLDER)
filelist = os.listdir(ORIGIN_DATA)
for index, file in enumerate(filelist):
    print("processing {0}, {1} out of {2}".format(file, index, len(filelist)))
    if not file.endswith(".png"):
        continue
    name = file[:file.rfind('_')]
    subFolder = os.path.join(OUT_FOLDER, name)
    if not os.path.exists(subFolder):
        os.mkdir(subFolder)
    copyfile(os.path.join(ORIGIN_DATA, file), os.path.join(subFolder, file))
    
