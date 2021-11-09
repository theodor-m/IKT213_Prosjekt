import numpy as np
import tensorflow as tf
new_model = tf.keras.models.load_model('trained-model.h5')
print(new_model.summary())
