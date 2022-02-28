
import os


BASE_PATH = "C:/Users/Nikola/Desktop/skin_lesions"
IMAGES_PATH = os.path.sep.join([BASE_PATH, "images"])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, "annotations_modified.csv"])

BASE_OUTPUT = "C:/Users/Nikola/Desktop/skin_lesions/outputs"

MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.h5"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_FILENAMES = os.path.sep.join([BASE_OUTPUT, "test_images.txt"])



INIT_LR = 1e-4
NUM_EPOCHS = 3
BATCH_SIZE = 32


