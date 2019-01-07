# -*- coding: utf-8 -*- 
from face_frontalization.frontalize_lfw import frontalize_lfw
import os

dataset = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "lfw")
outDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result", "face_frontal_lfw")

frontalize_lfw(dataset, outDir)