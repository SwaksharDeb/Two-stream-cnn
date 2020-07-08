from tensorflow.keras import backend as K
#from tensorflow.keras.engine.topology import Layer
from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Permute, Reshape
import numpy as np


class Transformer(Layer):

    def __init__(self, d_k, frames, **kwargs):
        #super(Transformer, self).__init__()
        self.d_k = d_k
        # self.scalar = np.sqrt(self.d_k)
        self.frames = frames
        super().__init__(**kwargs)

    def build(self, input_shape): # build the custom layer
        self.W = self.add_weight(name='transformer',
                                 shape=[input_shape[2], self.d_k],
                                 initializer='glorot_uniform',
                                 trainable=True)
        super().build(input_shape)

    def call(self, inputs):  # forward propagation

        inputs = K.permute_dimensions(inputs, (0, 3, 1, 2))
        inputs = K.reshape(inputs, (-1, self.frames*3, int(inputs.shape[3])))
        inputs = K.dot(inputs, self.W)
        inputs = K.reshape(inputs, (-1, 3, self.frames, self.d_k))
        inputs = K.permute_dimensions(inputs, (0, 2, 3, 1))
        # q, k, v = inputs
        # A = K.batch_dot(inputs, inputs, axes=[3, 3])/self.scalar
        # A = K.softmax(A)
        # A = K.batch_dot(A, inputs, axes=[3, 3])
        # A = K.permute_dimensions(A, (0, 2, 3, 1))
        return inputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.d_k, input_shape[3])

frames = 30
d_k = 30
input_shape = (frames, 25, 3)    
x = Input(shape=(frames, 25, 3))
x_a = Transformer(d_k, frames)(x)

up_0 = Input(shape=(frames, 25, 3))
up_1 = Input(shape=(frames, 25, 3))
down_0 = Input(shape=(frames, 25, 3))
down_1 = Input(shape=(frames, 25, 3))

fc_1 = Dense(4, activation=tf.nn.relu)(x)
fc_4 = Dense(units=60, activation='softmax', use_bias=True)(fc_1)

network = tf.keras.Model(inputs=[up_0, up_1, down_0, down_1], outputs=fc_4)

network = tf.keras.Model(inputs=x, outputs=fc_4)

#inputs = tf.keras.Input(shape=(3,))
inputs = Input(shape=(frames,25,3))
x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

