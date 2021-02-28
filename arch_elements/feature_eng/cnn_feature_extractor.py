from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Activation, Dropout, Dense

class CNNFeatureExtractor:
    """
    Convolutional Neural Network to extract features from images to be
    used in a Machine Learning algoritm.
    """
    def __init__(self, input_shape):
        model = Sequential()
        model.add(Conv2D(16,(5,5),padding='valid',input_shape = input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding = 'valid'))
        model.add(Dropout(0.4))
        model.add(Conv2D(32,(5,5),padding='valid'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding = 'valid'))
        model.add(Dropout(0.6))
        model.add(Conv2D(64,(5,5),padding='valid'))
        model.add(Activation('relu'))
        model.add(Dropout(0.8))
        model.add(Flatten())
        model.add(Dense(2))
        model.add(Activation('softmax'))
        self.transformer = Model(inputs=model.input,outputs=model.get_layer('dense_1').output)
        
    def transform(self, images):
        return self.transformer.predict(images)