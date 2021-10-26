"""
@author: Group 30; D. Kowalska, O. Luzon, X. Llani, T. Middleton

dataset: Best authors of all Time
source: Kaggle.com

Description: Dataset includes paintings from 50 different artists.
"""

import numpy as np
import os
import cv2 as cv
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import PIL
import PIL.Image
import tensorflow as tf
import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

train_images = []
train_labels = []
shape = (500, 500)

for artists in tqdm(os.listdir('Artists')):
    print(artists)
    for filename in tqdm(os.listdir('Artists/' + artists + '/train')):
        # Splitting images and storing image label into list
        img = cv.imread(os.path.join('Artists/' + artists + '/train/', filename))

        train_labels.append(artists.replace('_', ' '))

        img = cv.resize(img, shape)

        train_images.append(img)

train_labels = pd.get_dummies(train_labels).values

train_images = np.array(train_images)

x_train, x_val, y_train, y_val = train_test_split(train_images, train_labels, random_state=1)

print('Execution complete')

test_images = []
test_labels = []
for artists in tqdm(os.listdir('Artists')):
    print(f'Test Artist: {artists}')
    for filename in tqdm(os.listdir('Artists/' + artists + '/test')):
        img = cv.imread(os.path.join('Artists/' + artists + '/test/', filename))

        test_labels.append(artists.replace('_', ' '))

        img = cv.resize(img, shape)

        test_images.append(img)

test_images = np.array(test_images)

print('Execution complete: Test images added')


