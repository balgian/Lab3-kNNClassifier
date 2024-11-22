import sys
import numpy as np
import KNNClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Load the MNIST dataset
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    k: list[int]

    wine = load_wine()
    print(wine.DESCR)

    data: np.ndarray = wine.data
    target: np.ndarray = wine.target

    x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.2, random_state=42)

    # Normalize train_x and test_x
    x_train = (x_train - x_train.min()) / (x_train.max() - x_train.min())  # Normalize train_x
    x_test = (x_test - x_test.min()) / (x_test.max() - x_test.min())  # Normalize test_x

    # Values of k to test
    k = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50]
    num_mandatory_args = 5  # List of all mandatory arguments
    num_received_args = len([x_train, y_train, x_test, y_test] + list(k))  # List of all arguments
    if num_received_args < num_mandatory_args:
        raise ValueError("All arguments must be provided")
    # assert 0 <= k <= train_x.shape[0], "k must be between 0 and the number of training samples" 6000
    # assert 0 <= len(k) <= len(train_x[0]), "k must be between 0 and the number of training samples"  # 28
    for i in range(len(k)):
        assert 0 <= k[i] <= x_train.shape[0], "k must be between 0 and the number of training samples"
    # Accuracy results storage
    accuracy_results: dict[int, list[float]] = {neighbors: [] for _, neighbors in enumerate(k)}
    for _, neighbors in enumerate(k):
        # Create a 3-Nearest Neighbors classifier
        kNN: KNNClassifier = KNNClassifier.KNNClassifier(neighbors)
        # Train the kNN classifier
        kNN.fit(x_train, y_train)
        # Classify the test set according to the kNN rule
        y_pred = kNN.predict(x_test)
        accuracy_results[neighbors].append(float(np.mean(y_pred == y_test)))

    # Plot results
    plt.figure(figsize=(12, 8))
    plt.plot(accuracy_results.keys(), accuracy_results.values())
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.title('k-NN Accuracy for Each Digit vs. All Others')
    plt.grid(True)
    plt.show()
