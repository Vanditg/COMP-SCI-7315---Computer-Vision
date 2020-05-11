//==================================
// Computer Vision
// Student: Vandit Jyotindra Gajjar
// Student ID: a1779153
// Semester: 1
// Year: 2020
// Assignment: 1
//===================================

import cv2
import numpy as np

blocksize=4
searchrange=50

def computeNCC(block1, block2):
    v1 = block1.reshape(1, -1)
    v2 = block2.reshape(-1, 1)
    dot = np.dot(v1, v2)
    normv1 = np.linalg.norm(v1)
    normv2 = np.linalg.norm(v2)
    score = dot/(normv1 * normv2)
    if score==0:
        return 0
    else:
        return score

def BMNCC(img1, img2):
    h, w = img1.shape
    disparitymap = np.zeros(img1.shape)
    for i in range(h):
        for j in range(w):
            vl = img1[i-blocksize//2:i+blocksize//2+1, j-blocksize//2:j+blocksize//2+1]
            min_sim = 0
            for k in range(searchrange):
                vr = img2[i-blocksize//2:i+blocksize//2+1, j-blocksize//2-k:j+blocksize//2-k+1]
                if vl.shape == vr.shape:
                    score = computeNCC(vl, vr)
                    if score > min_sim:
                        min_sim = score
                else:
                    break
            disparitymap[i, j] = 1 - min_sim
    return disparitymap


img_l = cv2.imread("im0.png", 0)  
img_r = cv2.imread("im1.png", 0)  

disparity = BMNCC(img_l, img_r)  
disparity = cv2.normalize(src=disparity, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

cv2.imshow("disparity", img_l)
cv2.waitKey(0)
cv2.destroyAllWindows()