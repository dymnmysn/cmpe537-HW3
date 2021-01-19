import os
from scipy.spatial import distance
import numpy as np
from numpy.random import seed
from numpy.random import randint
# seed random number generator
seed(1)

def cent(k):
    centers=[]
    for i in range(k):
        centers.append(randint(0, 255, 128))
    return centers

def euclidian(p1,p2):
    return distance.euclidean(p1, p2)

def Kmeans(data,k,max_iterations):
    centers=cent(k)
    for m in range(max_iterations):
        print(m)
        clusters = [[] for _ in range(k)]
        for i in range(len(data)):
            distances=[euclidian((data[i]),(center)) for center in centers]
            class_ind = distances.index(min(distances))
            clusters[class_ind].append(data[i])
        for j in range(len(clusters)):
            if(len(clusters[j])>0):
                centers[j] = [round(sum(element) / len(clusters[j])) for element in zip(*clusters[j])]
            print(centers[j])
    return centers