import os
import cv2

import numpy as np

def feature_extraction_sift(dirName,labels,feature_vectors,keypoints):
    listOfFile = os.listdir(dirName)
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            getListOfFiles(fullPath,labels,feature_vectors,keypoints)
        elif(not (fullPath.startswith("Caltech20/training/.") or fullPath.startswith("Caltech20/testing/."))):
            label = dirName
            img = cv2.imread(fullPath)
            #resizing images to decrease the computational cost
            img = cv2.resize(img,(150, 150))
            #convert images to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #create SIFT object
            sift = cv2.xfeatures2d.SIFT_create(100)
            #detect SIFT features in both images
            keypoint, descriptor = sift.detectAndCompute(img, None)
            if (len(keypoint) >= 1): #to get rid of None's coming from sift keypoints
                feature_vectors.append(descriptor)
                keypoints.append(keypoint)
                labels.append(label)