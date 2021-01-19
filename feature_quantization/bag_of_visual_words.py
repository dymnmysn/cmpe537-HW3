import numpy as np

def get_histogram(class_name, image_features, centers):
    histogram = np.zeros(len(centers))
    for f in image_features:
        idx = np.argmin(map(lambda c: np.linalg.norm(c - f), centers))
        histogram[idx] += 1
    return histogram