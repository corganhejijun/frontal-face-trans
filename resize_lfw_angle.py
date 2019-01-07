from src.resize_for_train import getTrainImg

DATA_SET = "datasets/lfw_trans_angle"
OUT_DIR = "datasets/lfw_resize_for_train_128_angle"
MARGIN_FILE = "datasets/lfw_resize_margin_128_angle.txt"

IMAGE_SIZE = 128

getTrainImg(DATA_SET, IMAGE_SIZE, OUT_DIR, MARGIN_FILE) 