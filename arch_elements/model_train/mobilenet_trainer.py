from arch_elements.etl.transform import PrepareTrainingData
from arch_elements.model_def import Mobilenet
from . import AbstractTrainer
import tensorflow as tf
import os
import json

class MobilenetTrainer(AbstractTrainer):
    def __init__(self, pixels:int):
        self.pixels = pixels


    def __compile(self, metrics:list):
        num_classes = len(self.classes)
        self._model = Mobilenet(self.pixels, num_classes, do_fine_tuning = True)
        self._model.build((None,) + (self.pixels, self.pixels, 3,))
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9), 
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
            metrics=metrics)

    def train(self, training_data:PrepareTrainingData, epochs:int=60, metrics:list=['accuracy']) -> tf.keras.Model:
        self.classes = training_data.classes
        self.__compile(metrics)
        batch_size = training_data.batch_size
        self._history = self.model.fit(
            training_data.train_generator,
             epochs=epochs, steps_per_epoch=len(training_data.train_generator)-1,
            validation_data=training_data.valid_generator,
            validation_steps=len(training_data.valid_generator)-1
            ).history
        
        print("Evaluating on test data")
        results = self.model.evaluate(training_data.x_test, training_data.y_test, batch_size=batch_size)
        print("test loss, test acc:", results)

        return self.model

    @property
    def history(self):
        return self._history

    def save(self, model_name:str, path:str='.'):
        path = os.path.join(path, model_name)
        tf.saved_model.save(self.model, path)
        self.save_classes(path)

    def save_classes(self, model_path):
        with open(os.path.join(model_path, 'classes.json'), 'w') as f:
            classes = { i: c for i, c in enumerate(self.classes)}
            f.write(json.dumps(classes))

    def save_lite(self, model_name:str, path:str='.'):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        open(os.path.join(path, model_name + ".tflite"), "wb").write(tflite_model)

    def save_all(self, model_name:str, path:str='.'):
        self.save(model_name, path)
        self.save_lite(model_name, path)

    @property
    def model(self):
        return self._model

    def load(self, model_file:str, **kwargs):
        """Loads the model"""
        self._model = tf.keras.models.load_model(model_file, **kwargs)
        with open(os.path.join(model_file, 'classes.json'), 'r') as f:
            self.classes = json.load(f)