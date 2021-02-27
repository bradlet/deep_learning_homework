import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from time import time

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":

    data, labels = load_digits(return_X_y=True)

    n_samples, n_features = data.shape
    n_digits = np.unique(labels).size

    print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")

    k_means = KMeans(init="k-means++", n_clusters=n_digits, n_init=4, random_state=0)

    start_time = time()
    estimator = make_pipeline(StandardScaler(), k_means).fit(data)
    time_to_fit = time() - start_time

    results = ["k-means++", time_to_fit*1000, estimator[-1].inertia_]  # multiply by 1k to get ms instead of seconds
    print(results)
