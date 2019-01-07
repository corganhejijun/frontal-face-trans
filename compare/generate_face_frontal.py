# -*- coding: utf-8 -*- 
from face-frontalization.frontalize_lfw import frontalize_lfw

dataset = os.path.join(os.getcwd(), "..", "datasets", "lfw")
outDir = os.path.join("result", "face_frontal_lfw")

frontalize_lfw(dataset, outDir)