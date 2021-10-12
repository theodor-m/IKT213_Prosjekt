"""
@author: Group 30; D. Kowalska, O. Luzon, X. Llani, T. Middleton

dataset: Best authors of all Time
source: Kaggle.com

Description: Dataset includes paintings from 50 different artists.
"""
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mp_img
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import glob
import cv2 as cv
from keras.models import Sequential
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import normalize, to_categorical
Artists = ['Pablo_Picasso', 'Rene_Magritte', 'Vincent_van_Gogh']
img_dir = '~/datasets/modified/Artists/Lik mengde av bilder/'

# For iterating through folders
# for artist in Artists:
#     print(img_dir + artist + '.jpg')

t_img_dir = '\\datasets\\modified\\Artists\\Lik mengde av bilder\\Pablo_Picasso\\Pablo_Picasso_'

PATH_NAME = os.getcwd()

size = 250
new_dataset = []
for file in tqdm(range(1, 194, 1)):
    img = image.load_img(PATH_NAME + t_img_dir + str(file) +'.jpg', target_size=(size, size, 3))
    img = image.img_to_array(img)
    img = img/255
    new_dataset.append(img)

X = np.array(new_dataset)
y = ['Pablo Picasso']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
