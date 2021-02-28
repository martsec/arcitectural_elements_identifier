from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from collections.abc import Callable

class OneHot:
    def __init__(self):
        self.enc = OneHotEncoder(handle_unknown='ignore')
    
    def encode(self, labels:np.array) -> np.array:
        self.enc.fit(labels.reshape(-1, 1))
        labels_encoded = self.enc.transform(labels.reshape(-1, 1)).toarray()
        self.num_classes = self.classes.size
        return labels_encoded

    def get_class_string_from_index(self, index):
        return self.classes[index]

    def inverse_transform(self, y_label):
        return self.enc.inverse_transform(np.expand_dims(y_label, axis=0))[0][0]
    
    @property
    def classes(self):
        return self.enc.categories_[0]

class PrepareTrainingData:
    def __init__(self):
        self.extra_transformations = []

    def encode(self, y):
        self.encoder = OneHot() 
        return self.encoder.encode(y)

    @property
    def classes(self):
        return self.encoder.classes

    def transform(self, x:np.array, y:np.array, 
                  batch_size:int=64, test_split:float=0.2, validation_split:float=0.2, 
                  do_data_augmentation:bool=False, random_state:int=53, 
                  extra_transformations:list=[]) -> None:
        """
        Splits input data to train, test and validation.

        If `do_data_augmentation=True` it applies transformations on the train images 
        (rotation, flip, zoom...)

        :param extra_transformations: list of functions to apply to each image. 
                                      They are not applied if you get the data from the generator directly
        :type extra_transformations: list[Callable[np.array, np.array]]
        """
        self.batch_size = batch_size
        self.extra_transformations = extra_transformations
        self.__split(x, y, test_split=test_split, random_state=random_state)

        datagen_kwargs = dict(validation_split=validation_split)
        dataflow_kwargs = dict(batch_size=self.batch_size)

        
        valid_datagen = ImageDataGenerator(**datagen_kwargs)
        if validation_split != 0.:
            self.valid_generator = valid_datagen.flow(self.x_train, self.y_train,
                subset="validation", shuffle=True, **dataflow_kwargs)

        if do_data_augmentation:
            train_datagen = ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2, height_shift_range=0.2,
                shear_range=0.2, zoom_range=0.2,
                **datagen_kwargs)
        else:
            train_datagen = valid_datagen
        self.train_generator = train_datagen.flow(self.x_train, self.y_train,
            subset="training", shuffle=True, **dataflow_kwargs)
    
    def get_train_data(self) -> tuple:
        """Returns the training data as a python array tuple (x,y)"""
        return self.__get_from_generator(self.train_generator)
    
    def get_validation_data(self) -> tuple:
        """Returns the validation data as a python array tuple (x,y)"""
        return self.__get_from_generator(self.valid_generator)

    def get_test_data(self) -> tuple: 
        """Returns the test data as a python array typle (x,y)"""
        return (self.__extra_txs(self.x_test), self.y_test)

    def __get_from_generator(self, generator:ImageDataGenerator):
        x = []
        y = []
        generator.reset()
        for _ in range(len(generator)):
            ix, iy = generator.next()
            x.append(self.__extra_txs(ix))
            y.append(iy)
        return (np.concatenate(x), np.concatenate(y))

    def __extra_txs(self, x: np.array) -> np.array:
        # TODO find a more performant way. 
        # Tried np.apply_over_axes but I did not managed to get working
        if not self.extra_transformations:
            return x
        else: 
            for transform in self.extra_transformations:
                res = [transform(i) for i in x]
            return np.array(res)

    def __split(self, x, y, test_split, random_state):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split, random_state=random_state)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test