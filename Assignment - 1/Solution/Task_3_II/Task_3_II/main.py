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
import functions.Opts as Opts
import functions.Node as Node
import functions.Utils as Utils
from functions.Constants import INFINITY,MAX_ITERS

def main():
	PairWiseCostMatrix = Utils.PairWiseCostMatrix(opts.maxdisp)
	imgLeft, imgRight = Utils.ReturnImages(opts.imageno)
	M = imgLeft.shape[0]
	N = imgLeft.shape[1]
	
	AllVariableNodes = []
	instance = Node.UnaryFactorNode(imgLeft,imgRight,opts.maxdisp) 
	Utils.FillVarList(AllVariableNodes, instance.message, M, N, opts)
	
	HorFactorNodes = []
	VerFactorNodes = []
	Utils.FillFacLists(HorFactorNodes, VerFactorNodes, AllVariableNodes, M, N, PairWiseCostMatrix, opts)
	
	GlobalVarNodes = [item for sublist in AllVariableNodes for item in sublist]
	GlobalFacNodes = [item for sublist in HorFactorNodes for item in sublist] + [item for sublist in VerFactorNodes for item in sublist]

	for i in range(MAX_ITERS):
		print("Iteration no. %d starting..."%(i+1))
		if (not Utils.Propogate(GlobalVarNodes)):
			break
		
		SmootheningCost = sum([x.Loss() for x in GlobalFacNodes])
		DataCost = sum([x.Loss() for x in GlobalVarNodes])
		print("Iteration no. %d completed ==> Smoothening Cost : %d, Data Cost : %d, Total Cost : %d, Aversage Disparity : %f"%(i+1, SmootheningCost, DataCost, SmootheningCost + DataCost,sum([x.disparity for x in GlobalVarNodes])/len([x.disparity for x in GlobalVarNodes])))
		name = str(opts.imageno) + "_" + str(opts.smoothing) + "_" + str(i+1)
		Utils.SaveImage(GlobalVarNodes, M, N, name)

	SmootheningCost = sum([x.Loss() for x in GlobalFacNodes])
	DataCost = sum([x.Loss() for x in GlobalVarNodes])

	print("----------Complete-----------")
	name = str(opts.imageno) + "_" + str(opts.smoothing) + "_" + 'final'
	Utils.SaveImage(GlobalVarNodes, M, N, name, True)
	return

if __name__ == '__main__':
	opts = Opts.opts()
	main()