from tensorrt.lite import Engine
from tensorrt.infer import LogSeverity
import tensorrt
from random import randint
from PIL import Image
import numpy as np
import pandas as pd


data_test = pd.read_csv('test3.csv')

data_test = np.array(data_test.iloc[:, 1:])
print(data_test)
data_test = data_test.reshape(data_test.shape[0], 1, 28, 28)
data_test = data_test.astype('float32')
data_test /= 255

argmax = lambda res: np.argmax(res.reshape(10))

cases = []
labels = []

cases.append(data_test) # Append the image to list of images to process
labels.append(0) # Append the correct answer to compare later

# Create a runtime engine from plan file using TensorRT Lite API 
engine_single = Engine(PLAN="test_engine.engine",
                       postprocessors={"dense_2/Softmax":argmax})

results = []

for image in cases:
    result = engine_single.infer(image) # Single function for inference
    results.append(result)

print(results)
