from src.resize_for_train import getTrainImg

DATA_SET = "datasets/celeba_trans"
OUT_DIR = "datasets/celeba_resize_for_train"
MARGIN_FILE = "datasets/celeba_resize_margin.txt"

IMAGE_SIZE = 256

getTrainImg(DATA_SET, IMAGE_SIZE, OUT_DIR, MARGIN_FILE) 