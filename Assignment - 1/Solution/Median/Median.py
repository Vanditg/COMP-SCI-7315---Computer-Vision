//==================================
// Computer Vision
// Student: Vandit Jyotindra Gajjar
// Student ID: a1779153
// Semester: 1
// Year: 2020
// Assignment: 1
//===================================

import cv2 as cv

disp_img = cv.imread('Sword/disparity_1.png')
median_disp = cv.medianBlur(disp_img,5)

cv.imshow('med', median_disp)
cv.imwrite('Median_Sword/med_dis_1.png', median_disp)
cv.waitKey(0)
cv.destroyAllWindows()