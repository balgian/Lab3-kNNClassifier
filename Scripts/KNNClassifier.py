from typing import Any

import numpy as np
from numpy import floating


class KNNClassifier:
    def __init__(self, k: int = 3) -> None:
        self.x_train: np.ndarray = np.array([])
        self.y_train: np.ndarray = np.array([])
        self.k: int = k

        self.x_test: np.ndarray = np.array([])

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x_train = x
        self.y_train = y

    def predict(self, x: np.ndarray) -> np.ndarray:
        predictions = []
        for i in range(x.shape[0]):
            distances = np.linalg.norm(self.x_train - x[i], axis=1)
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            prediction = np.bincount(k_labels).argmax()
            predictions.append(prediction)
        return np.array(predictions)