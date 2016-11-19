import cv2
import numpy as np
import computing
# TODO make class


class Card:
	colours_count_array = []
	isRibbon = None		# 0 - RED, 1 - BLUE

	def __init__(self, name_of_file):
		self.name_of_file = name_of_file
