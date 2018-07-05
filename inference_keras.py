from random import randint
from PIL import Image
import numpy as np
import pandas as pd
import keras
from keras.models import Model
from keras.layers import Input, Dense
import tensorflow as tf

data_test = pd.read_csv('test3.csv')

data_test = np.array(data_test.iloc[:, 1:])
print(data_test)
data_test = data_test.reshape(data_test.shape[0], 28, 28, 1)
data_test = data_test.astype('float32')
data_test /= 255

model = tf.contrib.keras.models.load_model('FashionMnist.h5')

prediction = model.predict(data_test, batch_size=None, verbose=0)
print(prediction)
prediction = np.argmax(prediction, axis = 1)
print(prediction)