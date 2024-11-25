import sys
import numpy as np
import KNNClassifier
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    num_classes = max(y_true.max(), y_pred.max()) + 1
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
    return cm


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    return np.diag(cm) / np.sum(cm, axis=0)


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    return np.diag(cm) / np.sum(cm, axis=1)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    precision_val = np.nanmean(precision(y_true, y_pred))
    recall_val = np.nanmean(recall(y_true, y_pred))
    denominator = precision_val + recall_val
    if denominator == 0:
        return 0
    return 2 * (precision_val * recall_val) / denominator


if __name__ == "__main__":
    # Load the MNIST dataset
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    k: list[int]

    wine = load_wine()

    data: np.ndarray = wine.data
    target: np.ndarray = wine.target

    x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.2, random_state=42)

    # Normalize train_x and test_x
    train_min = x_train.min(axis=0)
    train_max = x_train.max(axis=0)
    # Normalize train_x
    x_train = (x_train - train_min) / (train_max - train_min)
    # Normalize test_x
    x_test = (x_test - train_min) / (train_max - train_min)

    # Values of k to test
    k = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50]
    num_mandatory_args = 5  # List of all mandatory arguments
    num_received_args = len([x_train, y_train, x_test, y_test] + list(k))  # List of all arguments
    if num_received_args < num_mandatory_args:
        raise ValueError("All arguments must be provided")

    classes = np.unique(y_train)
    for i in range(len(k)):
        assert 0 <= k[i] <= x_train.shape[0], "k must be between 0 and the number of training samples"
        assert isinstance(k[i], int), "k must be an integer"
    # Accuracy results storage
    accuracy_results: dict = {cls: {k_val: [] for k_val in k} for cls in classes}

    mean_val_prec: list = []
    median_val_prec: list = []
    avd_val_prec: list = []

    mean_val_rec: list = []
    median_val_rec: list = []
    avd_val_rec: list = []

    mean_val_f1: list = []
    median_val_f1: list = []
    avd_val_f1: list = []

    first_quantile_prec: list = []
    third_quantile_prec: list = []

    first_quantile_rec: list = []
    third_quantile_rec: list = []

    first_quantile_f1: list = []
    third_quantile_f1: list = []

    for cls in classes:
        # Create one-vs-all labels for training and testing
        binary_y_train = np.where(y_train == cls, 1, 0)
        binary_y_test = np.where(y_test == cls, 1, 0)

        precision_val: list = []
        recall_val: list = []
        f1_score_val: list = []

        for neighbors in k:
            # Create a k-NN classifier
            kNN = KNNClassifier.KNNClassifier(k=neighbors)
            # Train on binary labels
            kNN.fit(x_train, binary_y_train)
            # Predict on test set
            y_pred: np.ndarray
            error_rate: float
            y_pred, error_rate = kNN.predict(x_test, y_test)
            print(f"Class {cls}, k={neighbors}, Error rate: {round(100 * error_rate, 2)}%")

            # Compute accuracy
            accuracy = np.mean(y_pred == binary_y_test)
            accuracy_results[cls][neighbors].append(accuracy)

            # Compute confusion matrix
            conf_matrix: np.ndarray = confusion_matrix(binary_y_test, y_pred)
            # Compute precision, recall, and F1 score
            precision_val.append(precision(binary_y_test, y_pred))
            recall_val.append(recall(binary_y_test, y_pred))
            f1_score_val.append(f1_score(binary_y_test, y_pred))

        # Compute statistical measures of precision, recall, and F1 score
        mean_val_prec.append(np.mean(precision_val))
        median_val_prec.append(np.median(precision_val))
        avd_val_prec.append(np.mean(np.abs(np.array(precision_val) - mean_val_prec[-1])))
        first_quantile_prec.append(np.percentile(precision_val, 25))
        third_quantile_prec.append(np.percentile(precision_val, 75))

        mean_val_rec.append(np.mean(recall_val))
        median_val_rec.append(np.median(recall_val))
        avd_val_rec.append(np.mean(np.abs(np.array(recall_val) - mean_val_rec[-1])))
        first_quantile_rec.append(np.percentile(recall_val, 25))
        third_quantile_rec.append(np.percentile(recall_val, 75))

        mean_val_f1.append(np.mean(f1_score_val))
        median_val_f1.append(np.median(f1_score_val))
        avd_val_f1.append(np.mean(np.abs(np.array(f1_score_val) - mean_val_f1[-1])))
        first_quantile_f1.append(np.percentile(f1_score_val, 25))
        third_quantile_f1.append(np.percentile(f1_score_val, 75))

        # Print table for each class
        headers = ["Metric", "Mean", "Median", "AAD", "1st Quantile", "3rd Quantile"]
        table_data = [
            ["Precision", mean_val_prec[-1], median_val_prec[-1], avd_val_prec[-1], first_quantile_prec[-1],
             third_quantile_prec[-1]],
            ["Recall", mean_val_rec[-1], median_val_rec[-1], avd_val_rec[-1], first_quantile_rec[-1],
             third_quantile_rec[-1]],
            ["F1 Score", mean_val_f1[-1], median_val_f1[-1], avd_val_f1[-1], first_quantile_f1[-1],
             third_quantile_f1[-1]]
        ]
        print(f"Class {cls} Statistics over all k values:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Print total table
    # Print total table
    headers = ["Metric", "Mean", "Median", "AAD", "1st Quantile", "3rd Quantile"]
    total_data = [
        ["Precision", np.mean(mean_val_prec), np.median(mean_val_prec), np.mean(avd_val_prec),
         np.percentile(mean_val_prec, 25), np.percentile(mean_val_prec, 75)],
        ["Recall", np.mean(mean_val_rec), np.median(mean_val_rec), np.mean(avd_val_rec),
         np.percentile(mean_val_rec, 25), np.percentile(mean_val_rec, 75)],
        ["F1 Score", np.mean(mean_val_f1), np.median(mean_val_f1), np.mean(avd_val_f1), np.percentile(mean_val_f1, 25),
         np.percentile(mean_val_f1, 75)]
    ]
    print("Total Statistics for all classes:")
    print(tabulate(total_data, headers=headers, tablefmt="grid"))

    # Plot results
    plt.figure(figsize=(12, 8))
    for cls in classes:
        plt.plot(k, [100 * np.mean(acc) for acc in accuracy_results[cls].values()], label=f'Class {cls}', alpha=0.4,
                 marker='o', markersize=5, linewidth=2)
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy (%)')
    plt.title('k-NN Accuracy for Each Class vs. All Others')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.15)
    plt.xticks(k)
    plt.show()
