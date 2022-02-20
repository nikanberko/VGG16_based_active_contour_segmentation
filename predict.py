# import the necessary packages

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import mimetypes
import argparse
import imutils
import cv2
import os

# construct the argument parser and parse the arguments
import config
'''
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="C:/Users/Nikola/Desktop/skin_lesions/images/ISIC_0024505.jpg")
args = vars(ap.parse_args())
# determine the input file type, but assume that we're working with
# single input image
filetype = mimetypes.guess_type(args["input"])[0]
imagePaths = [args["input"]]
# if the file type is a text file, then we need to process *multiple*
# images
if "text/plain" == filetype:
    # load the filenames in our testing file and initialize our list
    # of image paths
    filenames = open(args["input"]).read().strip().split("\n")
    imagePaths = []
    # loop over the filenames
    for f in filenames:
        # construct the full path to the image filename and then
        # update our image paths list
        p = os.path.sep.join([config.IMAGES_PATH, f])
        imagePaths.append(p)
'''
# load our trained bounding box regressor from disk
print("[INFO] loading object detector...")
model = load_model(config.MODEL_PATH)
# loop over the images that we'll be testing using our bounding box
# regression model
imagePath="C:/Users/Nikola/Desktop/skin_lesions/images/ISIC_0024504.jpg"
# load the input image (in Keras format) from disk and preprocess
# it, scaling the pixel intensities to the range [0, 1]
image = load_img(imagePath, target_size=(224, 224))
image = img_to_array(image) / 255.0
image = np.expand_dims(image, axis=0)
# make bounding box predictions on the input image
preds = model.predict(image)[0]
(rx, ry, cx, cy) = preds
# load the input image (in OpenCV format), resize it such that it
# fits on our screen, and grab its dimensions
image = cv2.imread(imagePath)
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]
# scale the predicted bounding box coordinates based on the image
# dimensions
rx = int(rx * w)
ry = int(ry * h)
cx = int(cx * w * 1)
cy = int(cy * h * 1)
# draw the predicted bounding box on the image
cv2.ellipse(image, (rx, ry), (cx, cy), 0, 0, 360, (0, 0, 255), 5)
# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
