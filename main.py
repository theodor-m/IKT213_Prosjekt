"""
@author: Group 30; D. Kowalska, O. Luzon, X. Llani, T. Middleton

dataset: Best authors of all Time
source: Kaggle.com

Description: Dataset includes paintings from 50 different artists.
"""
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mp_img

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import normalize, to_categorical

