import glob
import cv2
import os
import re
import numpy as np
from collections import namedtuple


# { name_of_file : [month, pkt] }
facts_dictionary = {}
DecisionStruct = namedtuple("DesisionStruct", "name_of_file month pkt")
DecisionStruct.__new__.__defaults__ = (None, 0, 0)


def read_information(folder):
	files_list = glob.glob (folder + "*.jpg")
	for file in files_list:
		file = os.path.basename(file)
		if file[0].isdigit():
			file_atribiutes = file.split('.')[0].split('-')[:2]
			file_atribiutes[0] = int(float(file_atribiutes[0]))
			file_atribiutes[1] = int(float(re.findall(r'\d+', file_atribiutes[1])[0]))
			facts_dictionary[file] = file_atribiutes


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