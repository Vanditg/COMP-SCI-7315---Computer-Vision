//==================================
// Computer Vision
// Student: Vandit Jyotindra Gajjar
// Student ID: a1779153
// Semester: 1
// Year: 2020
// Assignment: 1
//===================================

import argparse

def opts():
	parser = argparse.ArgumentParser()
	parser.add_argument('-smoothing', type=int, default=10, help='Smoothening regularizing factor')
	parser.add_argument('-maxdisp', type=int, default=50, help='Maximum disparity allowed for a pixel')
	parser.add_argument('-imageno', type=int, default=1, help='Image number [1,2,3]')
	return parser.parse_args()