/==================================
// Computer Vision
// Student: Vandit Jyotindra Gajjar
// Student ID: a1779153
// Semester: 1
// Year: 2020
// Assignment: 2
//===================================

import numpy as np
import matplotlib.pyplot as plt
f = 50
tx, ty = 5, 10
v = np.array([ [1, 1,-1,-1], 
               [-1,-1,-1,-1], 
               [3, 1, 3, 1], 
               [1, 1, 1, 1]])
P = np.array([ [f, 0, 0, 0], 
               [0, f, 0, 0], 
               [0, 0, 1, 0]]) 
T = np.array([ [1, 0, 0, tx], 
               [0, 1, 0, ty],
               [0, 0, 1, 1],
               [0, 0, 0, 1]])
		
i = np.matmul(P, v)
i = i[0:2, :]/i[2]
j = np.matmul(T, v)
j = j[0:2, :]/j[2]
#fig = plt.figure()
#plt.scatter(i[0, :], i[1, :], c = 'r', marker = 's')
#plt.show()
fig = plt.figure()
plt.scatter(j[0, :], j[1, :], c = 'r', marker = 's')
plt.show()