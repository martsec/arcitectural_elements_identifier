import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from collections.abc import Iterable

class SVMClassifier: 
    def __init__(self, n_components: int):
        self._model = Pipeline([('pca', PCA(n_components=n_components)),
                                 ('clf',SVC(C=100,kernel='rbf',gamma=0.1))
                                ])

    def train(self, x: Iterable, y: Iterable):
        self._model.fit(x, y)

    def predict(self, x: Iterable) -> Iterable:
        return self._model.predict(x)