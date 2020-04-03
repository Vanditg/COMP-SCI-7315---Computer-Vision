//==================================
// Computer Vision
// Student: Vandit Jyotindra Gajjar
// Student ID: a1779153
// Semester: 1
// Year: 2020
// Assignment: 1
//===================================

import numpy as np
from functions.Constants import INFINITY

class UnaryFactorNode(object):

	def __init__(self, imgLeft, imgRight, MAX_DISPARITY):
		super(UnaryFactorNode, self).__init__()
		self.imgLeft = imgLeft
		self.imgRight = imgRight
		self.MAX_DISPARITY = MAX_DISPARITY

	def message(self, x, y):
		message = np.ones(self.MAX_DISPARITY)*INFINITY
		for i in range(self.MAX_DISPARITY):
			if (y < i):
				break
			message[i] = abs(int(self.imgLeft[x, y]) - int(self.imgRight[x, y-i]))
		return message

class PairWiseFactorNode(object):

	def __init__(self, MAX_DISPARITY, smoothening, pairwiseCostMatrix):
		super(PairWiseFactorNode, self).__init__()
		self.messages = [np.zeros(MAX_DISPARITY), np.zeros(MAX_DISPARITY)]
		self.disparities = [None,None]
		self.smoothening = smoothening
		self.pairwiseCostMatrix = pairwiseCostMatrix

	def updateMessage(self, index, message):
		self.messages[index] = message
	
	def updateDisp(self, index, disparity):
		self.disparities[index] = disparity
	
	def sendMessage(self, index):
		
		temp = np.ones(self.pairwiseCostMatrix.shape[0])*self.smoothening
		if ((self.disparities[0] is not None) and (self.disparities[1] is not None)):
			temp[self.disparities[1-index]] = 0
		else:
			temp[:] = 0
		return temp

	def Loss(self):
		return self.smoothening*self.pairwiseCostMatrix[self.disparities[0]][self.disparities[1]]

class VariableNode(object):

	def __init__(self, x, y, UnaryMessageFunc, MAX_DISPARITY):
		super(VariableNode, self).__init__()
		self.belief = None
		self.facnodes = []
		self.x = x
		self.y = y
		self.disparity = None
		self.MAX_DISPARITY = MAX_DISPARITY
		self.unarybelief = UnaryMessageFunc(self.x,self.y)

	def pushfac(self, fac):
		self.facnodes.append(fac)

	def buildBelief(self):
		self.resetBelief()
		self.messages = []
		for fac,facindex in (self.facnodes):
			self.messages.append(fac.sendMessage(facindex))
			self.belief += self.messages[-1]
		self.belief += self.unarybelief
		newDisp = np.argmin(self.belief)
		if self.disparity == newDisp:
			return False
		else :
			self.disparity = newDisp
			for i,(fac,facindex) in enumerate(self.facnodes):
				fac.updateDisp(facindex, newDisp)
			return True

	def sendMessages(self):
		for i,(fac,facindex) in enumerate(self.facnodes):
			fac.updateMessage(facindex, self.belief - self.messages[i])

	def resetBelief(self):
		self.belief = np.zeros(self.MAX_DISPARITY)

	def Loss(self):
		return self.unarybelief[self.disparity]