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

    def predict(self, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray | tuple[np.ndarray, float]:
        predictions = []
        for i in range(x.shape[0]):
            distances = np.linalg.norm(self.x_train - x[i], axis=1)
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            prediction = np.bincount(k_labels).argmax()
            predictions.append(prediction)
        if y is None:
            return np.array(predictions)
        return np.array(predictions), np.mean(predictions != y)
