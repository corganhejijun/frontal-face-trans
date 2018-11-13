from src.resize_for_train import getTrainImg

DATA_SET = "datasets/lfw_trans"
OUT_DIR = "datasets/lfw_resize_for_train_128"
MARGIN_FILE = "datasets/lfw_resize_margin_128.txt"

IMAGE_SIZE = 128

getTrainImg(DATA_SET, IMAGE_SIZE, OUT_DIR, MARGIN_FILE) 