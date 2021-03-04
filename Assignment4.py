import numpy as np
from time import time

from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


digits = load_digits(return_X_y=True)
data, labels = digits

n_samples, n_features = data.shape
n_digits = np.unique(labels).size

print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")


def row_accuracy(single_class_preds):
    row = np.zeros((10,))
    for pred in single_class_preds:
        row[pred] += 1
    return row


if __name__ == "__main__":
    # idxs == 'indices'
    idxs = labels.argsort()
    sorted_data, sorted_labels = data[idxs], labels[idxs]
    print("Pre-Processing: ", sorted_data.shape)

    processed_data = np.empty((64,))
    for i in range(0, 10):
        single_class_idxs = np.nonzero(sorted_labels == i)
        single_class = sorted_data[single_class_idxs]
        processed_data = np.vstack((processed_data, single_class[:100]))

    processed_data = np.delete(processed_data, 0, axis=0)  # Have to remove uninitialized first row from np.empty
    print("Processed: ", processed_data.shape)

    k_means = KMeans(init="k-means++", n_clusters=n_digits, n_init=4, random_state=0)

    start_time = time()
    estimator = make_pipeline(StandardScaler(), k_means).fit(processed_data)
    time_to_fit = time() - start_time
    print("Time to fit (ms): ", time_to_fit*1000)

    predictions = estimator[-1].labels_
    accuracy_table = row_accuracy(predictions[:100])
    for i in range(1, 10):
        accuracy_table = np.vstack((accuracy_table, row_accuracy(predictions[i*100:(i+1)*100])))

    print(accuracy_table)
