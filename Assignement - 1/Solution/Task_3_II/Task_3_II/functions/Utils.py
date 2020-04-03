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
from functions.Node import UnaryFactorNode, PairWiseFactorNode, VariableNode

def PairWiseCostMatrix(MAX_DISPARITY):
	pairwiseCostMatrix = np.ones((MAX_DISPARITY,MAX_DISPARITY))
	pairwiseCostMatrix[range(MAX_DISPARITY),range(MAX_DISPARITY)] = (0)
	return pairwiseCostMatrix

def ReturnImages(imageNum):
	imgLeft = cv2.imread('inputs/im0.png')
	imgLeft = cv2.cvtColor(imgLeft, cv2.COLOR_BGR2GRAY)
	imgRight = cv2.imread('inputs/im1.png')
	imgRight = cv2.cvtColor(imgRight, cv2.COLOR_BGR2GRAY)
	return imgLeft,imgRight

def FillVarList(AllVariableNodes, UnaryFunction, M, N, opts):
	for i in range(M):
		temp = []
		for j in range(N):
			temp.append(VariableNode(i,j,UnaryFunction,opts.maxdisp))
		AllVariableNodes.append(temp)
	return

def FillFacLists(HorFactorNodes, VerFactorNodes, AllVariableNodes, M, N, pairwiseCostMatrix, opts):
	for i in range(M-1):
		temp1 = []
		temp2 = []
		for j in range(N-1):
			
			temp1.append(PairWiseFactorNode(opts.maxdisp, opts.smoothing, pairwiseCostMatrix))
			AllVariableNodes[i][j].pushfac([temp1[-1],0])
			AllVariableNodes[i][j+1].pushfac([temp1[-1],1])
			
			temp2.append(PairWiseFactorNode(opts.maxdisp, opts.smoothing, pairwiseCostMatrix))
			AllVariableNodes[i][j].pushfac([temp2[-1],0])
			AllVariableNodes[i+1][j].pushfac([temp2[-1],1])
		HorFactorNodes.append(temp1)
		VerFactorNodes.append(temp2)

	temp1 = []
	temp2 = []

	for j in range(N-1):
		i = M-1
		temp1.append(PairWiseFactorNode(opts.maxdisp, opts.smoothing, pairwiseCostMatrix))
		AllVariableNodes[i][j].pushfac([temp1[-1],0])
		AllVariableNodes[i][j+1].pushfac([temp1[-1],1])

	for i in range(M-1):
		j = N-1
		temp2.append(PairWiseFactorNode(opts.maxdisp, opts.smoothing, pairwiseCostMatrix))
		AllVariableNodes[i][j].pushfac([temp2[-1],0])
		AllVariableNodes[i+1][j].pushfac([temp2[-1],1])

	HorFactorNodes.append(temp1)
	VerFactorNodes.append(temp2)
	return


def Propogate(GlobalVarNodes):
	change = False
	for node in GlobalVarNodes:
		change = node.buildBelief() or change
	for node in GlobalVarNodes:
		node.sendMessages()
	return change

def SaveImage(GlobalVarNodes, M, N, name, final=False):
	disparityMap = (np.asarray([(x.disparity) for x in GlobalVarNodes]).reshape(M,N))
	#disparityMap = 255*disparityMap/disparityMap.max()
	disparityMap = 3*disparityMap
	disparityMap = (disparityMap).astype(np.uint8)
	cv2.imwrite('outputs/' + str(name) + '.jpg', disparityMap)
	if (final):
		print(disparityMap)
	return