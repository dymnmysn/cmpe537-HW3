import numpy as np

def get_histogram(image_features, centers):
    histogram = np.zeros(len(centers))
    for f in image_features:
        idx = np.argmin(list(map(lambda c: np.linalg.norm(f - c), centers)))
        histogram[idx] += 1
    return histogram