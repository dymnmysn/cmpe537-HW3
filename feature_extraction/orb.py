import numpy as np
import cv2

class ORB:

    @staticmethod
    def compute_file(im_file_name):
        """
        Calculate and return ORB keypoints and
        descriptors of a given image file.

        @return
            tuple (kp, des)   kp is keypoints
                              des is descriptors
        """
        img = cv2.imread(im_file_name)
        orb = cv2.ORB_create()

        kp = orb.detect(img, None)

        kp, des = orb.compute(img, kp)
        
        return kp, des