import glob
import os
import re
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

