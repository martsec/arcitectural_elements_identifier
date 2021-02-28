from abc import ABC, abstractmethod 
from collections.abc import Iterable
from arch_elements.etl.transform import PrepareTrainingData
from arch_elements.utils import plot_confusion_matrix
import numpy as np

class AbstractTrainer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train(self, training_data:PrepareTrainingData):
        """Trains the model from the data stored in the TrainingData class"""

    @abstractmethod
    def save(self, model_name:str, path:str='.'):
        """Saves the model in the file system."""

    @abstractmethod
    def load(self, model_file:str, **kwargs):
        """Loads the model"""

    @property
    @abstractmethod
    def history(self):
        """Returns a dictionary with the metrics saved in the training run"""

    @property
    @abstractmethod
    def model(self):
        """Returns the model"""

    def evaluate(self, x:Iterable):
        return self.model.predict(x)

    def evaluate_metrics(self, x:Iterable, true_labels:Iterable, metrics:dict) -> dict:
        """Returns the evaluation of model for the given metrics"""
        predicted_labels = self._get_labels(self.evaluate(x))
        true_labels = self._get_labels(true_labels)
        return { m.__name__: m(predicted_labels, true_labels, **mkwargs) for m, mkwargs in metrics.items() }

    def plot_confusion_matrix(self, x:Iterable, true_labels:Iterable, 
                              label_names:list, **kwargs):
        predicted_labels = self._get_labels(self.evaluate(x))
        true_labels = self._get_labels(true_labels)
        plot_confusion_matrix(true_labels, predicted_labels, label_names, **kwargs)

    def plot_all_metrics(self):
        """Plots all the metrics stored. Will plot training and validation metrics in the same plot"""
        self.__check_if_history_initialized()
        metrics = set([m.replace('val_', '') for m in  self.history])
        num_metrics = len(metrics)
        import matplotlib.pyplot as plt
        _, axs = plt.subplots(num_metrics, 1, figsize=(15,num_metrics*6))
        for i, m in enumerate(metrics):
            label_t = ['training']
            label_v = ['validation']
            try:
                label_t= [label_t[0] + '_'+ str(i) for i in range(len(self.history[m][0]))]
                label_v= [label_v[0] + '_'+ str(i) for i in range(len(self.history[m][0]))]
            except:
                pass
            axs[i].plot(self.history[m])
            axs[i].plot(self.history['val_' + m])
            axs[i].set_title(m)
            axs[i].legend( label_t + label_v)

        plt.xlabel('Training steps')

    def plot_metric(self, metric_name:str):
        """Plots the metric """
        self.__check_if_history_initialized()
        import matplotlib.pylab as plt
        plt.figure()
        plt.ylabel(metric_name + " (training and validation)")
        plt.xlabel("Training Steps")
        #plt.ylim([0,2])
        plt.plot(self.history[metric_name])
        plt.plot(self.history["val_" + metric_name])

    def save_metrics(self, filename:str):
        self.__check_if_history_initialized()
        import json
        with open(filename + '.json', 'w') as f:
            f.write(json.dumps(self.history))

    def __check_if_history_initialized(self):
        if not self.history:
            raise ReferenceError('Tying to use train history but either you have not trained' + 
                                 ' the model yet or the model does not support it.')

    def _get_labels(self, labels: Iterable) -> Iterable:
        """
        Returns a 1D array containing the corresponding label ID for each entry. 
        Useful for getting the individual labels from the prediction output
        """
        if len(labels.shape) > 1:
             labels = np.argmax(labels, axis=1)
        return labels