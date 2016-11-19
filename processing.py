import cv2
import glob
import os
import numpy as np
from matplotlib import pyplot as plt
import computing
import argparse




def picture_processing(file_path):
	# READ IMAGE:
	filename = os.path.join(os.getcwd(), file_path)
	original_image = cv2.imread(filename, cv2.IMREAD_COLOR)
	image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

	# PROCESSING:
	# find ribbon:
	(colours_array, is_ribbon) = find_ribbon(original_image, os.path.basename(file_path))

	# DECISION:
	# TODO fill structure
	summary = computing.DecisionStruct(os.path.basename(file_path), colours_array, isRibbon=is_ribbon)
	make_decision(summary)


def find_ribbon(image, file_name):
	results_array, masks_array = computing.find_colour_count(image, file_name)
	ribbon_colour = None
	for n, mask in enumerate(masks_array[:2]):
		# TODO one more ribbon
		# red mask
		height, width = image.shape[:2]
		red_picture = np.zeros(image.shape, np.uint8)
		red_picture[:] = (0, 0, 255)
		red_mask = cv2.bitwise_and (red_picture, red_picture, mask=mask)

		# edges
		edges = cv2.Canny (red_mask, threshold1=100, threshold2=600, apertureSize=5)

		# contours of ribbon
		(_, cnts, _) = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
		ribbon = None
		for c in cnts:
			if cv2.contourArea(c) > 3000 and cv2.arcLength(c, closed=True) > 200:
				if tuple(c[c[:,:,1].argmin()][0])[1] > (height/3) or \
						tuple(c[c[:,:,1].argmax()][0])[1] < (height*2/3):
					continue
				ribbon = c
				ribbon_colour = n	# n <- number of colour
				print(file_name, "ribbon, colour: ", str(n))
				break;

		# SAVE:
		cv2.drawContours (red_mask, [ribbon], -1, (0, 255, 0), 3)
		computing.save_file (file_name, "ribbons_contours/" + str (n) + '/', red_mask)

	return results_array, ribbon_colour


def make_decision(result_of_processing):
	# TODO compare results and make a decision
	# COMPARE:
	a = 3
	# print(result_of_processing.name_of_file, "No decision")


def read_pictures ():
	files_list = sorted(glob.glob (computing.proces_input + "*.jpg"))
	for n, file in enumerate(files_list):
		# if n == 20:
		# 	break
		if os.path.basename(file)[0].isdigit():
			picture_processing(file)


if __name__ == '__main__':
	computing.read_information(computing.info_input)
	computing.compute_parameters()
	read_pictures()


# http://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html (7 dział - prostokąt)