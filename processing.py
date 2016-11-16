import cv2
import glob
import os
import numpy as np
from matplotlib import pyplot as plt
import information

name = ''


def make_decision(image, image_name):
	my_decision = information.DecisionStruct(name_of_file=image_name, month=5, pkt=2)

	# CHECK DECISION:
	if my_decision.name_of_file == None:
		print ("DECISION", image_name, "\tNO DECISION")
		return False
	print("DECISION\t", my_decision.name_of_file, my_decision.month, my_decision.pkt)
	for card in information.facts_dictionary.items():
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
	image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)	# cv2.IMREAD_GRAYSCALE / IMREAD_COLOR

	# PROCESSING:
	edges = cv2.Canny(image, 200, 900)	# smallest <- edge linking, largest <- initial segments of strong edges

	# SAVE FILES:
	output_path = os.path.join (os.getcwd(), "output/")
	if not os.path.exists(output_path):
		os.mkdir(output_path)
	new_path = os.path.join(output_path, os.path.basename(file_name) + name)
	cv2.imwrite(new_path, image)
	new_path = os.path.join (output_path, os.path.basename (file_name) + "e.jpg" + name)
	cv2.imwrite (new_path, edges)

	# DECISION:
	make_decision(edges, os.path.basename(filename))


def read_pictures ():
	files_list = glob.glob ("pictures/*.jpg")
	for n, file in enumerate(files_list):
		if n == 5:
			break
		if os.path.basename(file)[0].isdigit():
			picture_processing(file)


if __name__ == '__main__':
	information.read_information('pictures/')
	read_pictures()