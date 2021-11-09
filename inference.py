import numpy as np
from keras.preprocessing import image
from tensorflow.python.keras.backend import argmax
from tensorflow.python.keras.saving.save import load_model
import cv2 as cv
import os

model = load_model("./model")

test_images = []
test_labels = []
shape = (64, 64)
allArtists = []
for artists in os.listdir('Artists'):
    print(f'Test Artist: {artists}')
    allArtists.append(artists.replace('_', ' '))
    for filename in os.listdir('Artists/' + artists + '/test'):
        img = cv.imread(os.path.join('Artists/' + artists + '/test/', filename))

        test_labels.append(artists.replace('_', ' '))

        img = cv.resize(img, shape)

        test_images.append(img)

test_images = np.array(test_images)



# load and resize image to 64x64
test_image = image.load_img("Artists/Vincent_Van_Gogh/test/Vincent_Van_Gogh_174.jpg", target_size=(64,64))

# convert image to numpy array
imageToPredict = image.img_to_array(test_image)
# expand dimension of image
imageToPredict = np.expand_dims(imageToPredict, axis=0)
# making prediction with model
predictions = model.predict(imageToPredict)

classes = np.argmax(predictions, axis = 1)

print(allArtists)
print("Prediction of artist for image:" + "\n" + allArtists[int(classes)])


