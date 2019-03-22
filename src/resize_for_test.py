import os
import cv2

folder = 'datasets/celeb_train/val_test'
destFolder = 'datasets/celeb_train/resized_val_test'
if not os.path.isdir(destFolder):
    os.mkdir(destFolder)
for file in os.listdir(folder):
    print("processing " + file)
    filePath = os.path.join(folder, file)
    img = cv2.imread(filePath)
    resizedImg = cv2.resize(img, (256, 128))
    destPath = os.path.join(destFolder, file)
    cv2.imwrite(destPath, resizedImg)