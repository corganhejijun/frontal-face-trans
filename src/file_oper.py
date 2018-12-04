# -*- coding: utf-8 -*- 
import os
from shutil import copyfile

def copy_celeba(identity, dataset, outDir):
    file = open(identity, "r")
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    count = 0
    for line in file:
        result = line.split(' ')
        fileName = result[0]
        identity = result[1]
        subDir = os.path.join(outDir, identity[:-1])
        if not os.path.exists(subDir):
            os.mkdir(subDir)
        dest = os.path.join(subDir, fileName)
        src = os.path.join(dataset, fileName)
        if not os.path.exists(dest) and os.path.exists(src):
            copyfile(src, dest)
        count += 1
        if count % 1000 == 0:
            print("%d files had been copied" % count)
    file.close()