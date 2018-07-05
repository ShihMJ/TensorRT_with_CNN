from random import randint
from PIL import Image
import numpy as np
import pandas as pd
import keras
from keras.models import Model
from keras.layers import Input, Dense
import tensorflow as tf
import os
from tensorflow.python.platform import gfile


data_test = pd.read_csv('test3.csv')

data_test = np.array(data_test.iloc[:, 1:])
print(data_test)
data_test = data_test.reshape(data_test.shape[0], 28, 28, 1)
data_test = data_test.astype('float32')
data_test /= 255
"""
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(data_test, W) + b)

feed_dict = {data_test: [your_image]}
classification = tf.run(y, feed_dict)
print(classification)
"""
with tf.Graph().as_default() as graph: # Set default graph as graph

           with tf.Session() as sess:
                # Load the graph in graph_def
                print("load graph")

                # We load the protobuf file from the disk and parse it to retrive the unserialized graph_drf
                with gfile.FastGFile("FashionMnist.pb",'rb') as f:

                                # Set FCN graph to the default graph
                                graph_def = tf.GraphDef()
                                graph_def.ParseFromString(f.read())
                                sess.graph.as_default()

                                # Import a graph_def into the current default Graph (In this case, the weights are (typically) embedded in the graph)

                                tf.import_graph_def(
                                graph_def,
                                input_map=None,
                                return_elements=None,
                                name="",
                                op_dict=None,
                                producer_op_list=None
                                )

                                # Print the name of operations in the session
                                """
                                for op in graph.get_operations():
                                        print("Operation Name :",op.name)         # Operation name
                                        print("Tensor Stats :",str(op.values()))     # Tensor name
								"""
                                # INFERENCE Here
                                l_input = graph.get_tensor_by_name('conv2d_1_input:0') # Input Tensor
                                l_output = graph.get_tensor_by_name('output_node0:0') # Output Tensor

                                print("Shape of input : ", tf.shape(l_input))
                                #initialize_all_variables
                                tf.global_variables_initializer()

                                # Run Kitty model on single image
                                Session_out = sess.run( l_output, feed_dict = {l_input : data_test} )
                                print(Session_out)
                                Session_out = np.argmax(Session_out, axis = 1)
                                print(Session_out)