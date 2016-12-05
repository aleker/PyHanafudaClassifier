import cv2
import glob
import os
import numpy as np
from matplotlib import pyplot as plt
import computing
import card
import argparse


def picture_processing(file_path):
    # TODO fill class
    # READ IMAGE:
    filename = os.path.join(os.getcwd(), file_path)
    original_image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    card_list = []

    current_card = card.Card(os.path.basename(file_path))

    # PROCESSING:
    # find rectangle (card contour):
    findCardShape(original_image, os.path.basename(file_path))

    # compute hu moments:

    # find ribbon:
    #(current_card.colours_count_array, current_card.isRibbon) = findRibbon(original_image, os.path.basename(file_path))

    # DECISION:
    makeDecision(current_card)


def is_close(beginning, ending):
    boundary = 30
    if (beginning[0] > ending[0] - (beginning[0] > (ending[0] + boundary))).all:
        return False
    elif (beginning[0] < ending[0] - (beginning[0] + boundary < ending[0])).all:
        return False
    elif (beginning[1] > ending[1] - (beginning[1] > (ending[1] + boundary))).all:
        return False
    elif (beginning[1] < ending[1] - (beginning[1] + boundary < ending[1])).all:
        return False
    else:
        return True


def findCardShape(colour_image, filename):
    gray_image = cv2.cvtColor(colour_image, cv2.COLOR_BGR2GRAY)
    # SCALING:
    height, width = gray_image.shape[:2]
    max_width = 500.0
    if height > width:
        coefficient = height / max_width
        new_height = int(max_width)
        new_width = int(width / coefficient)
    else:
        coefficient = width / max_width
        new_width = int(max_width)
        new_height = int(height / coefficient)
    gray_image = cv2.resize(gray_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    colour_image = cv2.resize(colour_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    # CANNY:
    image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(image, threshold1=500, threshold2=1100, apertureSize=5)
    kernel = np.ones((1,1), np.uint8)
    opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((10, 10), np.uint8)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    edge_dilation = closing

    # CONTOURS:
    (_, cnts, hierarchy) = cv2.findContours(edge_dilation.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:]
    cards = []
    for n,c in enumerate(cnts):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > 700:      # [Next, Previous, First_child, Parent] hierarchy[0][n][3] == -1
            cards.append(c)
    # SAVING:
    # computing.save_file('e' + filename, 'testing_output/', edge_dilation)
    cv2.drawContours(colour_image, cards, -1, (0, 255, 0), 3)
    computing.save_file(filename, '1_find_shape_of_card/', colour_image)
    # print(filename, len(cnts), len(cards))


def findRibbon(image, file_name):
    org = image
    results_array, masks_array = computing.find_colour_count(image, file_name)
    ribbon_colour = None
    for n, mask in enumerate(masks_array[:2]):
        # TODO one more ribbon
        # red mask
        height, width = image.shape[:2]
        red_picture = np.zeros(image.shape, np.uint8)
        red_picture[:] = (0, 0, 255)
        red_mask = cv2.bitwise_and(red_picture, red_picture, mask=mask)

        # edges
        edges = cv2.Canny(red_mask, threshold1=100, threshold2=600, apertureSize=5)

        # contours of ribbon
        (_, cnts, _) = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
        ribbon = None
        for c in cnts:
            if cv2.contourArea(c) > 3000 and cv2.arcLength(c, closed=True) > 200:
                if tuple(c[c[:, :, 1].argmin()][0])[1] > (height / 3) or \
                                tuple(c[c[:, :, 1].argmax()][0])[1] < (height * 2 / 3):
                    continue
                ribbon = c
                ribbon_colour = n  # n <- number of colour
                # print (file_name, "ribbon, colour: ", str (n))
                break;

        # SAVE:
        cv2.drawContours(red_mask, [ribbon], -1, (0, 255, 0), 3)
        computing.save_file(file_name, "ribbons_contours/" + str(n) + '/', red_mask)

    return results_array, ribbon_colour


def makeDecision(card):
    # TODO compare results and make a decision
    # COMPARE:
    a = 3
# print(result_of_processing.name_of_file, "No decision")


def read_pictures():
    types = ('*.jpg', '*.JPG')
    files_list = []
    for type in types:
        files_list.extend(sorted(glob.glob(computing.test_input + type)))
    print('liczba plików do przetworzenia: ', len(files_list))
    for n, file in enumerate(files_list):
        # if n == 2:
        #  	break
        #if os.path.basename(file)[0].isdigit():
        picture_processing(file)


if __name__ == '__main__':
    # REFERENCE PICTURES:
    computing.read_information(computing.reference_input)
    computing.compute_parameters()

    # TESTED PICTURES:
    read_pictures()


    # http://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html (7 dział - prostokąt)
