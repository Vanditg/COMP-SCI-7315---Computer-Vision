//==================================
// Computer Vision
// Student: Vandit Jyotindra Gajjar
// Student ID: a1779153
// Semester: 1
// Year: 2020
// Assignment: 1
//===================================

import numpy as np
import cv2
import matplotlib.pyplot as plt

left = cv2.imread('images/im0.png')
right = cv2.imread('images/im1.png')

left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)


#cv2.imshow('Left', left_gray)
cv2.imwrite('left_gray.png', left_gray)
#cv2.imshow('Right', right_gray)
cv2.imwrite('right_gray.png', right_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

patch_loc = [100, 120]
patch_size = [30, 30]

patch_left = left_gray[patch_loc[0]:(patch_loc[0] + patch_size[0]),
                       patch_loc[1]:(patch_loc[1] + patch_size[1])]

strip_right = right_gray[patch_loc[0]:(patch_loc[0] + patch_size[0]), :]

#cv2.imshow('Image Patch Left', patch_left)
#cv2.imwrite('patch_left.png', patch_left)
#cv2.imshow('Image Strip Right', strip_right)
#cv2.imwrite('strip_right.png', strip_right)
cv2.waitKey(0)
cv2.destroyAllWindows()


def best_x(patch, strip):

    min_diff = float('inf')
    best_x = 0
    for x in range(0, strip.shape[1] - patch.shape[1] + 1):
        extracted_patch = strip[:, x:(x + patch.shape[1])]
        ssd = np.sum((patch - extracted_patch)**2)
        if ssd < min_diff:
            best_x = x
            min_diff = ssd

    return best_x

x = best_x(patch_left, strip_right)
patch_right = strip_right[:, x:(x + patch_left.shape[1])]

fig = plt.figure(figsize=(8, 6))
fig.canvas.set_window_title('Best X Patch Detection')
plt.subplot(311), plt.imshow(patch_left, 'gray')
plt.title('Original Patch from the Left Image'), plt.xticks([]), plt.yticks([])
plt.subplot(312), plt.imshow(strip_right, 'gray')
plt.title('Original Strip from the Right Image'), plt.xticks([]), plt.yticks([])
plt.subplot(313), plt.imshow(patch_right, 'gray')
plt.title('Extracted Patch from the Right Image'), plt.xticks([]), plt.yticks([])
cv2.imwrite('patch_right.png', patch_right)
plt.show()

y = 75
b = 1

left_strip = left_gray[y:(y + b), :]
right_strip = right_gray[y:(y + b), :]

#cv2.imshow('Left Disparity Image Strip', left_strip)
#cv2.imwrite('left_strip.png', left_strip)
#cv2.imshow('Right Disparity Image Strip', right_strip)
#cv2.imwrite('right_strip.png', right_strip)
cv2.waitKey(0)
cv2.destroyAllWindows()

def match_strips(left_strip, right_strip, block_size):
    num_blocks = left_strip.shape[1] // block_size 

    disparity = np.zeros(num_blocks)

    for i in range(num_blocks):
        x_left = i*block_size
        patch_left = left_strip[:, x_left:(x_left + block_size)] 
        x_right = best_x(patch_left, right_strip)
        disparity[i] = x_left - x_right

    return disparity

disparity = match_strips(left_strip, right_strip, b)
print (disparity)

fig = plt.figure(figsize=(8, 3))
fig.canvas.set_window_title('Stereo Disparity')
plt.subplot(311), plt.imshow(left_strip, 'gray')
plt.title('Strip from the Left Image'), plt.xticks([]), plt.yticks([])
plt.subplot(312), plt.imshow(right_strip, 'gray')
plt.title('Strip from the Right Image'), plt.xticks([]), plt.yticks([])
plt.subplot(313), plt.plot(disparity)
plt.title('Disparity Between the Two Images')

plt.show()