# -*- coding: utf-8 -*- 
import os
from scipy import misc
from PIL import Image
import numpy as np
import math

DATA_SET = "datasets\\lfw_trans\\Aaron_Peirsol"
OUT_DIR = "datasets\\lfw_resize_for_train"
IMAGE_SIZE = 256

for imgName in os.listdir(DATA_SET):
    img = misc.imread(os.path.join(DATA_SET, imgName))
    img_resize = misc.imresize(img, (256, 256))
    width = img.shape[1]
    height = img.shape[0]
    for i in range(height):
        for j in range(width):
            if (img[i, j] == [0, 0, 0]).all():
                img_resize[int(math.ceil(j * IMAGE_SIZE / width)), int(math.ceil(i * IMAGE_SIZE / height))] = [0, 0, 0]
                img_resize[int(math.ceil(j * IMAGE_SIZE / width)) + 1, int(math.ceil(i * IMAGE_SIZE / height))] = [0, 0, 0]
                img_resize[int(math.ceil(j * IMAGE_SIZE / width)) - 1, int(math.ceil(i * IMAGE_SIZE / height))] = [0, 0, 0]
                img_resize[int(math.ceil(j * IMAGE_SIZE / width)), int(math.ceil(i * IMAGE_SIZE / height) + 1)] = [0, 0, 0]
                img_resize[int(math.ceil(j * IMAGE_SIZE / width)), int(math.ceil(i * IMAGE_SIZE / height) - 1)] = [0, 0, 0]
    target = Image.new('RGB', (img_resize.shape[0]*2, img_resize.shape[1]))
    target.paste(Image.fromarray(np.uint8(img_resize)), (0, 0))
    target.paste(Image.fromarray(np.uint8(img_resize)), (img_resize.shape[0] + 1, 0))
    target.save(os.path.join(OUT_DIR, imgName))