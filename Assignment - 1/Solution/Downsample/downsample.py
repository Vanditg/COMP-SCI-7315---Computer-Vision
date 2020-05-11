//==================================
// Computer Vision
// Student: Vandit Jyotindra Gajjar
// Student ID: a1779153
// Semester: 1
// Year: 2020
// Assignment: 1
//===================================

import cv2
 
img_l = cv2.imread('Images/1/im1E.png', cv2.IMREAD_UNCHANGED)
img_r = cv2.imread('Images/1/im1L.png', cv2.IMREAD_UNCHANGED)
print('Original Dimensions : ',img_l.shape)
print('Original Dimensions : ',img_r.shape)

scale_percent = 20 # percent of original size

l_width = int(img_l.shape[1] * scale_percent / 100)
l_height = int(img_l.shape[0] * scale_percent / 100)
l_dim = (l_width, l_height)

r_width = int(img_r.shape[1] * scale_percent / 100)
r_height = int(img_r.shape[0] * scale_percent / 100)
r_dim = (r_width, r_height)

# resize image
l_resized = cv2.resize(img_l, l_dim, interpolation = cv2.INTER_AREA)
r_resized = cv2.resize(img_r, r_dim, interpolation = cv2.INTER_AREA)

cv2.imwrite('Output/1/im1E.png', l_resized)
cv2.imwrite('Output/1/im1L.png', r_resized)

print('Resized Dimensions : ',l_resized.shape)
print('Resized Dimensions : ',r_resized.shape)