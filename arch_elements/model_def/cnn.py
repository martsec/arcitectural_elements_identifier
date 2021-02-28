from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, InputLayer
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.regularizers import l2
import tensorflow as tf

class CNN(tf.keras.Model):
    """
    Implementation of a Convolutional Neural Network to classify images. 
    It uses a dropout layer with a variable rate set at initialization.

    :param pixels: Pixels of the square image
    :param num_classes: Number of classes to predict
    :param dropout: Dropout rate for the internal layers. Between 0 and 1. Default `0.2`
    :param normalize: if true, model will handle rescaling (normalization) of the pixel values
    """

    def __init__(self, pixels:int, num_classes:int, dropout=.2, normalize=True):
        super(CNN, self).__init__()
        self.input_l = InputLayer(input_shape=(pixels, pixels, 3))
        self.rescaling = Rescaling(1./255)
        self._normalize_input = normalize
        self.conv1 = Conv2D(16, 3, padding='same', activation='relu')
        self.pool1 = MaxPooling2D()
        self.conv2 = Conv2D(32, 3, padding='same', activation='relu')
        self.pool2 = MaxPooling2D()
        self.conv3 = Conv2D(64, 3, padding='same', activation='relu')
        self.pool3 = MaxPooling2D()
        self.dropout = Dropout(rate = dropout)
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation='relu')
        self.output_l = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.0001))

    def call(self, inputs, training=None):
        input_layer = self.input_l(inputs)
        h = self.conv1(input_layer)
        if self._normalize_input:
           h =  self.rescaling(h)
        h = self.pool1(h)
        h = self.dropout(h, training=training)
        h = self.conv2(h)
        h = self.pool2(h)
        h = self.dropout(h, training=training)
        h = self.conv3(h)
        h = self.pool3(h)
        h = self.dropout(h, training=training)
        h = self.flatten(h)
        h = self.dense1(h)
        output_layer = self.output_l(h)
        return output_layer
