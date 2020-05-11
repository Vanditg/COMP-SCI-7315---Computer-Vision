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
v = np.array([ [1, 1,-1,-1], 
               [1,-1,1,-1], 
               [1, 1, 1, 1], 
               [1, 1, 1, 1]])
P = np.array([ [f, 0, 0, 0], 
               [0, f, 0, 0], 
               [0, 0, 1, 0]])
i = np.matmul(P, v)
i = i[0:2, :]/i[2]
fig = plt.figure()
plt.scatter(i[0, :], i[1, :], c = 'r', marker = 's')
plt.show()