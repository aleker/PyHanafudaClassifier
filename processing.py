import cv2
import glob
import os
import numpy as np
from matplotlib import pyplot as plt
import computing
import argparse

name = ''


def make_decision(image, image_name):
	my_decision = computing.DecisionStruct(name_of_file=image_name, month=5, pkt=2)

	# CHECK DECISION:
	if my_decision.name_of_file == None:
		print ("DECISION", image_name, "\tNO DECISION")
		return False
	print("DECISION\t", my_decision.name_of_file, my_decision.month, my_decision.pkt)
	for card in computing.facts_dictionary.items():
		if card[0] == my_decision.name_of_file:		# card as list [key, [month, pkt]]
			print("FACT\t\t", card[0], card[1][0], card[1][1])
			if card[1][0] != my_decision.month:
				print("\t\tWrong month.")
			if card[1][1] != my_decision.pkt:
				print("\t\tWrong points.")
			break
	return my_decision


def picture_processing(file_name):
	# READ IMAGE:
	filename = os.path.join(os.getcwd(), file_name)
	original_image = cv2.imread(filename, cv2.IMREAD_COLOR)
	#image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)	# cv2.IMREAD_GRAYSCALE / IMREAD_COLOR
	image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

	#-------------------------------------------------
	# PROCESSING:
	res = np.hstack ((image, image))

	# blur: Bilateral filtering has the nice property of removing noise in
	# 		the image while still preserving the actual edges.
	#		it works similarly to beauty-filter
	blurred = cv2.bilateralFilter(image, 11, 17, 17)
	#blurred = cv2.GaussianBlur(image, (3,3), 0)
	res = np.hstack ((image, blurred))

	# clahe:
	# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	# cl1 = clahe.apply(image)
	# res = np.hstack((image,cl1))

	# equalization:
	# equ = cv2.equalizeHist(image)
	# res = np.hstack((image, equ))

	# canny: smallest <- edge linking, largest <- initial segments of strong edges
	# edges = cv2.Canny(image, threshold1=200, threshold2=600, apertureSize=3)
	edges = computing.auto_canny(blurred)

	# contours:
	(_, contours, _) = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key=cv2.contourArea, reverse=True)[:6]
	res = np.hstack ((original_image, cv2.drawContours(original_image.copy(), contours, -1, color=(0,255,0), thickness=3)))
	# http://www.pyimagesearch.com/2014/04/21/building-pokedex-python-finding-game-boy-screen-step-4-6/

	#hull = cv2.convexHull(edges)	# to z ręką http://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html


	#--------------------------------------------------
	# SAVE FILES:
	output_path = os.path.join (os.getcwd(), "output/")
	if not os.path.exists(output_path):
		os.mkdir(output_path)
	new_path = os.path.join(output_path, os.path.basename(file_name) + name)
	cv2.imwrite(new_path, res)
	new_path = os.path.join (output_path, os.path.basename (file_name) + "e.jpg" + name)
	cv2.imwrite (new_path, edges)

	# DECISION:
	decision = make_decision(edges, os.path.basename(filename))


def read_pictures ():
	files_list = glob.glob ("pictures/*.jpg")
	for n, file in enumerate(files_list):
		if n == 5:
			break
		if os.path.basename(file)[0].isdigit():
			picture_processing(file)


if __name__ == '__main__':
	computing.read_information('pictures/')
	read_pictures()

# http://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html (7 dział - prostokąt)