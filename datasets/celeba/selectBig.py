import os

folder = 'celeba_identified'
maxCount = 0
maxFolderName = ''
for file in os.listdir(folder):
    path = os.path.join(folder, file)
    if len(os.listdir(path)) > maxCount:
        maxCount = len(os.listdir(path))
        maxFolderName = file
        print("{} has {} files".format(file, maxCount))