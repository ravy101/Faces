import sys
import os
import numpy as np
import pandas as pd 
from tensorflow import keras
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator

model_name = 'stride_model.m5'
img_folder = 'video_faces/'

#Read command line arguments
if(len(sys.argv) > 2):
    model_name = sys.argv[1]
    img_folder = sys.argv[2]
else:
    print("Error: Missing command line arguments.")
    print("Usage: python user_existing.py <Model File> <Image Folder>")
    #sys.exit()

#load existing model
existing_model = keras.models.load_model(model_name)

#load and prepare test files
img_files = os.listdir(img_folder)
images = []
for file in img_files:
    try:
        images.append(img_to_array(load_img(img_folder + file, target_size=(128,128))))
    except:
        pass
test_images = np.array(images)
test_images = test_images * 1./255

#make predicitons and output
predictions = existing_model.predict_classes(test_images)
print(predictions == 1)