import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import cv2
from glob import glob
from PIL import Image

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split

# load the contents of the CSV annotations file
import config

print("[INFO] loading dataset...")
rows = open(config.ANNOTS_PATH).read().strip().split("\n")
# initialize the list of data (images), our target output predictions
# (bounding box coordinates), along with the filenames of the
# individual images
data = []
targets = []
filenames = []


# loop over the rows
for row in rows:
    #break the row into the filename and bounding box coordinates
    row = row.split(",")
    (filename, cx, cy, rx, ry) = row
    print(filename)

    # derive the path to the input image, load the image (in OpenCV
    # format), and grab its dimensions
    imagePath = os.path.sep.join([config.IMAGES_PATH, filename])
    image = cv2.imread(imagePath)
    (h, w) = image.shape[:2]
    # scale the bounding box coordinates relative to the spatial
    # dimensions of the input image
    cx = float(cx) / w
    cy = float(cy) / h
    rx = float(rx) / w
    ry = float(ry) / h

    # load the image and preprocess it
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    # update our list of data, targets, and filenames
    data.append(image)
    targets.append((cx, cy, rx, ry))
    filenames.append(filename)

# convert the data and targets to NumPy arrays, scaling the input
# pixel intensities from the range [0, 255] to [0, 1]
data = np.array(data, dtype="float32") / 255.0
print(filenames)
targets = np.array(targets, dtype="float32")

# partition the data into training and testing splits using 90% of
# the data for training and the remaining 10% for testing
split = train_test_split(data, targets, filenames, test_size=0.20, train_size=0.80,
                         random_state=42)
# unpack the data split
(trainImages, testImages) = split[:2]
(trainTargets, testTargets) = split[2:4]
(trainFilenames, testFilenames) = split[4:]
# write the testing filenames to disk so that we can use then
# when evaluating/testing our bounding box regressor
print("[INFO] saving testing filenames...")
f = open(config.TEST_FILENAMES, "w")
f.write("\n".join(testFilenames))
f.close()

# load the VGG16 network, ensuring the head FC layers are left off
vgg = VGG16(weights='imagenet', include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))
# freeze all VGG layers so they will *not* be updated during the
# training process
vgg.trainable = False
# flatten the max-pooling output of VGG
flatten = vgg.output
flatten = Flatten()(flatten)
# construct a fully-connected layer header to output the predicted
# bounding box coordinates
bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid")(bboxHead)
# construct the model we will fine-tune for bounding box regression
model = Model(inputs=vgg.input, outputs=bboxHead)


# initialize the optimizer, compile the model, and show the model
# summary
opt = Adam(lr=config.INIT_LR)
model.compile(loss="mse", optimizer=opt)
print(model.summary())
# train the network for bounding box regression
print("[INFO] training bounding box regressor...")
H = model.fit(
    trainImages, trainTargets,
    validation_data=(testImages, testTargets),
    batch_size=config.BATCH_SIZE,
    epochs=config.NUM_EPOCHS,
    verbose=1)

# serialize the model to disk
print("[INFO] saving object detector model...")
model.save(config.MODEL_PATH, save_format="h5")
# plot the model training history
N = config.NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config.PLOT_PATH)
'''
###########################################
# METHOD 1: Read files using file name from the csv and add corresponding
# image in a pandas dataframe along with labels.
# This requires lot of memory to hold all thousands of images.
# Use datagen if you run into memory issues.

skin_df = pd.read_csv('C:/Users/Nikola/Desktop/skin_lesions/annotations.csv')

# Now time to read images based on image ID from the CSV file
# This is the safest way to read images as it ensures the right image is read for the right ID

image_path = {os.path.splitext(os.path.basename(x))[0]: x
          for x in glob(os.path.join('C:/Users/Nikola/Desktop/skin_lesions/', '*', '*.jpg'))}
#print(image_path.items())
# Define the path and add as a new column

skin_df['path'] = skin_df['_name'].map(image_path.get)

# Use the path to read images.
skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((32, 32))))
#print(skin_df['image'])



#############################################################################
# Reorganize data into subfolders based on their labels
# then use keras flow_from_dir or pytorch ImageFolder to read images with
# folder names as labels

# Sort images to subfolders first

import shutil

# Dump all images into a folder and specify the path:
data_dir = os.getcwd() + "/data/all_images/"

# Path to destination directory where we want subfolders
dest_dir = os.getcwd() + "/data/reorganized/"

# Read the csv file containing image names and corresponding labels
skin_df2 = pd.read_csv('data/HAM10000/HAM10000_metadata.csv')
print(skin_df['dx'].value_counts())

label = skin_df2['dx'].unique().tolist()  # Extract labels into a list
label_images = []

# Copy images to new folders
for i in label:
os.mkdir(dest_dir + str(i) + "/")
sample = skin_df2[skin_df2['dx'] == i]['image_id']
label_images.extend(sample)
for id in label_images:
    shutil.copyfile((data_dir + "/" + id + ".jpg"), (dest_dir + i + "/" + id + ".jpg"))
label_images = []

'''
'''
# Now we are ready to work with images in subfolders

### FOR Keras datagen ##################################
# flow_from_directory Method
# useful when the images are sorted and placed in there respective class/label folders
# identifies classes automatically from the folder name.
# create a data generator

from keras.preprocessing.image import ImageDataGenerator
import os
from matplotlib import pyplot as plt

# Define datagen. Here we can define any transformations we want to apply to images
datagen = ImageDataGenerator()

# define training directory that contains subfolders
train_dir = os.getcwd() + "/data/reorganized/"
# USe flow_from_directory
train_data_keras = datagen.flow_from_directory(directory=train_dir,
                                           class_mode='categorical',
                                           batch_size=16,  # 16 images at a time
                                           target_size=(32, 32))  # Resize images

# We can check images for a single batch.
x, y = next(train_data_keras)
# View each image
for i in range(0, 15):
image = x[i].astype(int)
plt.imshow(image)
plt.show()
'''

