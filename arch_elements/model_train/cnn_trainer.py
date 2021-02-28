from arch_elements.etl.transform import PrepareTrainingData
from arch_elements.model_def import CNN
from . import AbstractTrainer
import tensorflow as tf
import time
import os
import json

class CNNTrainer(AbstractTrainer):
    """
    Trains the CNN model for 128x128x3 images
    """
    def __init__(self, pixels:int, **kwargs):
        self.pixels = pixels
        self.model_kwargs = kwargs

    def __compile(self, classes:list, metrics:list):
        num_classes = len(classes)
        self._model = CNN(self.pixels, num_classes, **self.model_kwargs)
        self._model.build((None,) + (self.pixels, self.pixels, 3,))
        self.model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
              metrics=metrics)

    def train(self, training_data:PrepareTrainingData, epochs:int=60, metrics:list=['accuracy']) -> tf.keras.Model:
        self.classes = training_data.classes
        self.__compile(self.classes, metrics)
        batch_size = training_data.batch_size
        start_time = time.time()
        self._history = self.model.fit(
            training_data.train_generator,
            epochs=epochs, steps_per_epoch=len(training_data.train_generator)-1,
            validation_data=training_data.valid_generator,
            validation_steps=len(training_data.valid_generator)-1,
            ).history
        print("--- Total time taken %s minutes ---" % ((time.time() - start_time)/60.))
        print("\n\n # Evaluating on test data")
        results = self.model.evaluate(training_data.x_test, training_data.y_test, batch_size=batch_size)
        for k, v in zip(['loss'] +  metrics, results):
            print(str(k) + ":", v)

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