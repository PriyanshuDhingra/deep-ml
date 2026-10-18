import numpy as np

def kmeans_plus_plus_init(X: np.ndarray, k: int, seed: int = None) -> np.ndarray:
    """
    Initialize k centroids using the K-Means++ algorithm.
    
    Args:
        X: Data points of shape (n_samples, n_features)
        k: Number of centroids to initialize
        seed: Random seed for reproducibility
    
    Returns:
        Centroids of shape (k, n_features)
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples, n_features = X.shape
    centroids = np.zeros((k, n_features))
    first_index = np.random.randint(n_samples)
    centroids[0] = X[first_index]
    chosen = [first_index]

    for i in range(1, k):
        distances = np.zeros(n_samples)

        for j, point in enumerate(X):
            nearest = float('inf')

            for centroid in centroids[:i]:
                sq_dist = sum((point[m] - centroid[m])**2 for m in range(n_features))
                nearest = min(nearest, sq_dist)

            distances[j] = nearest

        total_dist = distances.sum()

        if total_dist == 0:
            probabilities = np.ones(n_samples)
            probabilities[chosen] = 0
            probabilities /= probabilities.sum()
        else:
            probabilities = distances / total_dist

        next_index = np.random.choice(n_samples, p=probabilities)
        centroids[i] = X[next_index]
        chosen.append(next_index)

    return centroids