import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import os
import json

class TensorflowModelPredictor:
    def __init__(self, model_save_path:str):
        # TODO load classes
        m = tfa.metrics.F1Score(10)
        self.model = tf.keras.models.load_model(model_save_path, custom_objects={'f1_score': m})
        self._classes = self._load_classes(model_save_path)

    def _load_classes(self, model_save_path):
        with open(os.path.join(model_save_path, 'classes.json'), 'r') as f:
            cl = json.load(f)
        return cl

    def predict(self, x: np.array) -> np.array:
        """
        Predicts the best fit label and returns its string value
        """
        predictions = self.model.predict(x)
        get_label = lambda ps: self.classes[str(np.argmax(ps))]
        predicted_labels = np.apply_along_axis(get_label, 1, predictions)

        return predicted_labels

    @property
    def classes(self):
        return self._classes