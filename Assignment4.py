import numpy as np
from time import time

from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":

    digits = load_digits(return_X_y=True)
    data, labels = digits

    n_samples, n_features = data.shape
    n_digits = np.unique(labels).size

    print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")

    k_means = KMeans(init="k-means++", n_clusters=n_digits, n_init=4, random_state=0)

    start_time = time()
    estimator = make_pipeline(StandardScaler(), k_means).fit(data)
    time_to_fit = time() - start_time

    # results = ["k-means++", time_to_fit*1000, estimator[-1].inertia_]  # multiply by 1k to get ms instead of seconds
    # print(results)

    # idxs == 'indices'
    idxs = labels.argsort()
    sorted_data, sorted_labels = data[idxs], labels[idxs]
    print("Pre-Processing: ", sorted_data.shape)

    processed_data = np.empty((64,))
    for i in range(0, 10):
        single_class_idxs = np.nonzero(sorted_labels == i)
        single_class = sorted_data[single_class_idxs]
        print(single_class.shape)
        processed_data = np.vstack((processed_data, single_class[:100]))

    processed_data = np.delete(processed_data, 0, axis=0)  # Have to remove uninitialized first row from np.empty
    print("Processed: ", processed_data.shape)

