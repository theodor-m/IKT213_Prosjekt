import numpy as np
import argparse
import os
from keras.preprocessing import image
from tensorflow.python.keras.backend import argmax
from tensorflow.python.keras.saving.save import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

classNames = []

for artists in os.listdir('Artists'):
    classNames.append(artists.replace('_', ' '))


def classify(img_path):
    img = image.load_img(img_path, color_mode= "grayscale", target_size=(500, 500))
    img_array = image.img_to_array(img)

    img_batch = np.expand_dims(img_array, axis=0)

    model = load_model("./trained-model.h5", compile=True)
    prediction = model.predict(img_batch)

    print("################## Results of model inference ##################")
    print("Artist prediction:" + " " + classNames[prediction.argmax()])
    print("Probability distribution:")
    for x in classNames:
        print(x + ':' + " " + str(prediction[0][classNames.index(x)]))

parser = argparse.ArgumentParser()
parser.add_argument('file',
                    help="Enter file location.")
args = parser.parse_args()

classify(args.file)


