"""
@author: Group 30; D. Kowalska, O. Luzon, X. Llani, T. Middleton

dataset:
source dataset: Best authors of all Time
source: Kaggle.com

Description: Dataset includes paintings from 50 different artists.
"""

import numpy as np
import os
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from tensorflow.keras.models import Sequential

train_images = []
train_labels = []
shape = (64, 64)

for artists in os.listdir('Artists'):
    print(artists)
    for filename in os.listdir('Artists/' + artists + '/train'):
        # Splitting images and storing image label into list
        img = cv.imread(os.path.join('Artists/' + artists + '/train/', filename), 0)

        train_labels.append(artists.replace('_', ' '))

        img = cv.resize(img, shape)

        train_images.append(img)

train_labels = pd.get_dummies(train_labels).values

train_images = np.array(train_images)

x_train, x_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.18, random_state=1)

print('Execution complete')

test_images = []
test_labels = []
for artists in os.listdir('Artists'):
    print(f'Test Artist: {artists}')
    for filename in os.listdir('Artists/' + artists + '/test'):
        img = cv.imread(os.path.join('Artists/' + artists + '/test/', filename))

        test_labels.append(artists.replace('_', ' '))

        img = cv.resize(img, shape)

        test_images.append(img)

test_images = np.array(test_images)

print('Execution complete: Test images added')

model = Sequential()
model.add(Conv2D(kernel_size=(3, 3), filters=32, activation='relu', input_shape=(500, 500, 1,)))
model.add(Conv2D(kernel_size=(3, 3), filters=30, activation='relu'))
model.add(MaxPool2D(2, 2))
model.add(Conv2D(filters=30,kernel_size = (3,3),activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(filters=30,kernel_size = (3,3),activation='relu'))

model.add(Flatten())

model.add(Dense(20,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(3,activation = 'softmax'))

model.compile(
    loss='categorical_crossentropy',
    metrics=['acc'],
    optimizer='Adadelta'
)

model.summary()

history = model.fit(x_train, y_train, epochs=50, batch_size=11, validation_data=(x_val, y_val))
model.save("model", save_format="h5")

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# save the trained model
model.save('trained-model.h5', include_optimizer=False)
