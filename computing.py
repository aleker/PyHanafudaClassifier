import glob
import cv2
import os
import re
import numpy as np
from collections import namedtuple
from tempfile import TemporaryFile
from sklearn import tree
import pydotplus

import configuration


# { name_of_file : [month, pkt, [colours], [hu_moments_red], [hu_moments_blue], [hu_moments_white] }
facts_dictionary = {}

colour_boundaries = [  # ([B,G,R], [B,G,R])
    ([4, 9, 86], [80, 88, 220]),  # RED
    ([80, 31, 4], [220, 100, 60]),  # BLUE
    ([180, 160, 160], [255, 255, 255])
]


def read_information(folder):
    files_list = sorted(glob.glob(folder + "*.jpg"))
    for file in files_list:
        file = os.path.basename(file)
        if file[0].isdigit():
            file_atribiutes = file.split('.')[0].split('-')[:3]
            file_atribiutes[0] = int(float(file_atribiutes[0]))
            file_atribiutes[1] = int(float(re.findall(r'\d+', file_atribiutes[1])[0]))
            file_atribiutes[2] = int(float(re.findall(r'\d+', file_atribiutes[2])[0]))
            facts_dictionary[file] = file_atribiutes
    print('REFERENCE pictures count:', len(facts_dictionary))


def find_colour_count(image, file_name = None):
    colourful_count = []
    masks = []
    for n, (lower, upper) in enumerate(colour_boundaries):
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        mask = cv2.inRange(image, lower, upper)
        kernel = np.ones((3, 3), np.uint8)
        mask2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = mask2

        colourful_pixels_count = cv2.countNonZero(mask)
        colourful_count.append(colourful_pixels_count)
        masks.append(mask)

        if file_name is not None:
            # print(file_name, n, colourful_pixels_count)
            # SAVE MASK:
            image_with_mask = cv2.bitwise_and (image, image, mask=mask)
            save_file(file_name + configuration.name, "ribbons_masks/"+str(n)+"/", image_with_mask)

    return colourful_count, masks


def computeHuMoments(image, file_name = None):
    results_array, masks_array = find_colour_count(image, file_name)
    hu_moments_for_all_colours = []
    for n, mask in enumerate(masks_array[:]):
        # white mask
        height, width = image.shape[:2]
        red_picture = np.zeros(image.shape, np.uint8)
        red_picture[:] = (255, 255, 255)
        white_mask = cv2.bitwise_and(red_picture, red_picture, mask=mask)

        # hu moments - silhouette
        hu_moments = cv2.HuMoments(cv2.moments(cv2.cvtColor(white_mask, cv2.COLOR_BGR2GRAY))).flatten()
        hu_moments_for_all_colours.append(hu_moments)
        # print (file_name, n, "\n", hu_moments)

        # SAVE:
        if file_name is not None:
            save_file(file_name, "hu_moments/" + str(n) + '/', white_mask)

    return hu_moments_for_all_colours


def compute_parameters():
    for file_key in sorted(facts_dictionary.keys()):
        # READ FILE:
        filename = os.path.join(os.getcwd(), configuration.reference_input + file_key)
        original_image = cv2.imread(filename, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        # COMPUTE PARAMETERS:
        # 1. colour count:
        (array_of_colour_values, _) = find_colour_count(original_image, file_key)
        for coordinate in array_of_colour_values:
            facts_dictionary[file_key].append(coordinate)

        # 2. hu_moments:
        hu_moments_for_all_colours = computeHuMoments(original_image, file_key)
        for colour in hu_moments_for_all_colours:
            for moment in colour:
                facts_dictionary[file_key].append(moment)

    print("REFERENCE pictures computed.")


def createDecisionTree():
    samples = []
    labels = []
    for card_key in facts_dictionary.keys():
        features = []
        for characteristic in facts_dictionary[card_key][2:]:
            features.append(characteristic);
        samples.append(features)
        labels.append(card_key)
    clf_tree = tree.DecisionTreeClassifier()
    clf_tree = clf_tree.fit(samples, labels)

    with open(configuration.result_dir + "tree.dot", 'w') as f:
        f = tree.export_graphviz(clf_tree, out_file=f)
    os.unlink(configuration.result_dir + 'tree.dot')

    print("Decision Tree created.")
    return clf_tree


def save_file(file_name, output_folder, result):
    output_path = os.path.join(os.getcwd(), output_folder)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    new_path = os.path.join(output_path, file_name + configuration.name)
    cv2.imwrite(new_path, result)
