//==================================
// Computer Vision
// Student: Vandit Jyotindra Gajjar
// Student ID: a1779153
// Semester: 1
// Year: 2020
// Assignment: 1
//===================================

import argparse
import numpy as np
import cv2
import pylab as pl

def Normalised_Cross_Correlation(roi, target):

    cor = np.sum(roi * target)
    nor = np.sqrt( (np.sum(roi ** 2))) * np.sqrt(np.sum(target ** 2))

    return cor / nor

def template_matching(img, target):

    height, width = img.shape
    tar_height, tar_width = target.shape
    (max_Y, max_X) = (0, 0)
    MaxValue = 0

    img = np.array(img, dtype="int")
    target = np.array(target, dtype="int")
    NccValue = np.zeros((height-tar_height, width-tar_width))

    for y in range(0, height-tar_height):
        for x in range(0, width-tar_width):

            roi = img[y : y+tar_height, x : x+tar_width]

            NccValue[y, x] = Normalised_Cross_Correlation(roi, target)

            if NccValue[y, x] > MaxValue:
                MaxValue = NccValue[y, x]
                (max_Y, max_X) = (y, x)

    return (max_X, max_Y)


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "Path to input image")
    ap.add_argument("-t", "--target", required = True, help = "Path to target")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"], 0)
    new_image = cv2.resize(image, None, fx=0.4, fy=0.4, interpolation = cv2.INTER_CUBIC)
    cv2.imshow('new_image', 1)
    target = cv2.imread(args["target"], 0)

    height, width = target.shape

    top_left = template_matching(new_image, target)

    cv2.rectangle(new_image, top_left, (top_left[0] + width, top_left[1] + height), 0, 3)

    pl.subplot(111)
    pl.imshow(new_image)
    pl.title('result')
    pl.show()