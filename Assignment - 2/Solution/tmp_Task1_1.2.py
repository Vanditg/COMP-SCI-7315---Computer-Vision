/==================================
// Computer Vision
// Student: Vandit Jyotindra Gajjar
// Student ID: a1779153
// Semester: 1
// Year: 2020
// Assignment: 2
//===================================
import math as m
import numpy as np
import matplotlib.pyplot as plt
f = 50
tx, ty, tz = -1, 0, 0
t = 90
v = np.array([ [1, 1,-1,-1], 
               [-1,-1,-1,-1], 
               [3, 1, 3, 1], 
               [1, 1, 1, 1]])
P = np.array([ [f, 0, 0, 0], 
               [0, f, 0, 0], 
               [0, 0, 1, 0]]) 
T = np.array([ [1, 0, 0, tx], 
               [0, 1, 0, ty],
               [0, 0, 1, tz],
               [0, 0, 0, 1]])
R = np.array([[m.cos(t), 0, m.sin(t), 0], 
             [0, 1, 0, 0], 
             [-m.sin(t), 0, m.cos(t), 0], 
             [0, 0, 0, 1]])
		
i = np.matmul(P, v)
i = i[0:2, :]/i[2]
j = np.matmul(T, v)
j = j[0:2, :]/j[2]
k = np.matmul(T, R)
a = np.matmul(k, v)
a = a[0:2, :]/a[2]
#fig = plt.figure()
#plt.scatter(i[0, :], i[1, :], c = 'r', marker = 's')
#plt.show()
#fig = plt.figure()
#plt.scatter(j[0, :], j[1, :], c = 'r', marker = 's')
#plt.show()
fig = plt.figure()
plt.scatter(a[0, :], a[1, :], c = 'r', marker = 's')
plt.show()