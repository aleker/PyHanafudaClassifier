import cv2
import glob
import os
import numpy as np
from matplotlib import pyplot as plt


def picture_processing(file_name):
	filename = os.path.join(os.getcwd(), file_name)
	img = cv2.imread(filename, cv2.IMREAD_COLOR)	# cv2.IMREAD_GRAYSCALE

	cv2.imshow('image', img)
	cv2.waitKey (0)
	cv2.destroyAllWindows ()


def read_pictures ():
	files_list = glob.glob ("pictures/*.jpg")
	for file in files_list:
		if os.path.basename (file)[0].isdigit():
			picture_processing(file)
	#picture_processing("pictures/01-20pkt.jpg")


if __name__ == '__main__':
	read_pictures()