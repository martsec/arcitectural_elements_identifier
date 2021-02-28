from arch_elements.model_def import SVMClassifier
from arch_elements.model_train import AbstractTrainer
from arch_elements.etl.transform import PrepareTrainingData
import pickle
import os
import json
import numpy as np

class SVMCTrainer(AbstractTrainer):
    def __init__(self):
        self._model = SVMClassifier(n_components=100)

    def train(self, training_data:PrepareTrainingData):
        """Trains the model from the data stored in the TrainingData class"""
        self.classes = training_data.classes
        x_train, y_train = training_data.get_train_data()
        if len(y_train.shape) != 1:
            y_train = np.argmax(y_train, axis=1)
        self.model.train(x_train, y_train)

    def save(self, model_name:str, path:str='.'):
        """Saves the model in the file system."""
        file_name =  os.path.join(path, model_name + ".pkl")
        with open(file_name, 'wb') as f:
            pickle.dump(self.model, f)
        with open(file_name + 'classes.json', 'w') as f:
            classes = { i: c for i, c in enumerate(self.classes)}
            f.write(json.dumps(classes))

    def load(self, model_file:str):
        """Loads the model"""
        with open(model_file, 'rb') as f:
            self._model = pickle.load(f)
        with open(model_file + 'classes.json', 'r') as f:
            self.classes = json.load(f)

    @property
    def history(self):
        """This type of model does not support training history"""
        return None

    @property
    def model(self):
        """Returns the model"""
        return self._model