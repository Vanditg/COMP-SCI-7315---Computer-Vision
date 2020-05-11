/==================================
// Computer Vision
// Student: Vandit Jyotindra Gajjar
// Student ID: a1779153
// Semester: 1
// Year: 2020
// Assignment: 2
//===================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

f = 17
tx, ty = 0, 0
p1 = np.array([ [f, 0, 0, tx], 
                [0, f, 0, ty],
                [0, 0, 1, 0]])

p2 = np.eye(3, 4)
p2[0, 3] = -1

N = 5
point3d = np.empty((4, N), np.float32)
point3d[:3, :] = np.random.randn(3, N)
point3d[3, :] = 1

point = p1 @ point3d
point = point[:2, :] / point[2, :]
point[:2, :] += np.random.randn(2, N) * 1e-2

point_2 = p2 @ point3d
point_2 = point_2[:2, :] / point_2[2, :]
point_2[:2, :] += np.random.randn(2, N) * 1e-2

point3d_rec = cv2.triangulatePoints(p1, p2, point, point_2)
point3d_rec /= point3d_rec[3, :]

d = cdist(point3d, point3d_rec)
print('d =', d)

print(point3d[:3].T)
print(point3d_rec[:3].T)

fig = plt.figure()
plt.scatter(point3d_rec[0, :], point3d_rec[1, :], c = 'r', marker = 's')
plt.show()