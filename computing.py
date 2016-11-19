import glob
import cv2
import os
import re
import numpy as np
from collections import namedtuple


name = ''
info_input = 'pictures/'
proces_input = 'pictures/'

# { name_of_file : [month, pkt, [colours]}
facts_dictionary = {}
DecisionStruct = namedtuple("DecisionStruct", "name_of_file colours_amount_array isRibbon")
DecisionStruct.__new__.__defaults__ = (None, [], False)

colour_boundaries = [	# ([B,G,R], [B,G,R])
	([4, 9, 86], [80, 88, 220]),		# RED
	([80, 31, 4], [220, 100, 60]),		# BLUE
	([230, 230, 230], [255, 255, 255])	# WHITE
]


def read_information(folder):
	files_list = sorted(glob.glob (folder + "*.jpg"))
	for file in files_list:
		file = os.path.basename(file)
		if file[0].isdigit():
			file_atribiutes = file.split('.')[0].split('-')[:2]
			file_atribiutes[0] = int(float(file_atribiutes[0]))
			file_atribiutes[1] = int(float(re.findall(r'\d+', file_atribiutes[1])[0]))
			facts_dictionary[file] = file_atribiutes


def find_colour_count(image, file_name):
	colourful_count = []
	masks = []
	for n, (lower, upper) in enumerate (colour_boundaries):
		lower = np.array (lower, dtype="uint8")
		upper = np.array (upper, dtype="uint8")

		mask = cv2.inRange (image, lower, upper)
		kernel = np.ones((3,3), np.uint8)
		mask2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
		mask = mask2

		colourful_pixels_count = cv2.countNonZero (mask)
		colourful_count.append(colourful_pixels_count)
		masks.append(mask)
		# print(file_name, n, colourful_pixels_count)

		# SAVE MASK:
		# output = cv2.bitwise_and (image, image, mask=mask)
		# output_path = os.path.join (os.getcwd (), str (n) + "ribbon_output/")
		# if not os.path.exists (output_path):
		# 	os.mkdir (output_path)
		# new_path = os.path.join (output_path, file_name + name)
		# cv2.imwrite (new_path, output)

	return colourful_count, masks


def compute_parameters():
	for file_key in sorted(facts_dictionary.keys()):
		# READ FILE:
		filename = os.path.join (os.getcwd (), info_input + file_key)
		original_image = cv2.imread (filename, cv2.IMREAD_COLOR)
		image = cv2.cvtColor (original_image, cv2.COLOR_BGR2GRAY)

		# COMPUTE PARAMETERS:
		(array_of_colour_values, _) = find_colour_count(original_image, file_key)
		facts_dictionary[file_key].append(array_of_colour_values)


		# SAVE PARAMETERS TO DICTIONARY:


def auto_canny (image, sigma=0.33):
	# FROM http://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
	# This trick simply takes the median of the image, and then constructs upper and lower
	# thresholds based on a percentage of this median. In practice, sigma=0.33  tends to obtain good results.

	# compute the median of the single channel pixel intensities
	v = np.median (image)

	# apply automatic Canny edge detection using the computed median
	lower = int (max (0, (1.0 - sigma) * v))
	upper = int (min (255, (1.0 + sigma) * v))
	edged = cv2.Canny (image, lower, upper)

	# return the edged image
	return edged