import cv2
import glob
import os
import numpy as np
from matplotlib import pyplot as plt
import computing
import card
import argparse
import math
from scipy import ndimage

# cart, c4.5 w skimage

def picture_processing(file_path):
    # TODO fill class
    # READ IMAGE:
    filename = os.path.join(os.getcwd(), file_path)
    original_image = cv2.imread(filename, cv2.IMREAD_COLOR)

    # PROCESSING:
    # scale image (max width(height) is 500):
    original_image = scaleImage(original_image)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # find cards on picture:
    card_list = findCardShape(original_image.copy(), os.path.basename(file_path))

    for n, one_card_contour in enumerate(card_list):
        current_card = card.Card(str(n) + os.path.basename(file_path))
        # take out card_picture from image:
        card_picture = cropCardFromPicture(one_card_contour, original_image.copy())
        # compute hu moments:
        current_card.huMoments = computing.computeHuMoments(card_picture.copy())
        # find ribbon:
        (current_card.colours_count_array, current_card.isRibbon) = findRibbon(card_picture, str(n) + os.path.basename(file_path))

        # DECISION:
        makeDecision(current_card)


def scaleImage(image):
    # SCALING:
    height, width = image.shape[:2]
    max_width = 500.0
    if height > width:
        coefficient = height / max_width
        new_height = int(max_width)
        new_width = int(width / coefficient)
    else:
        coefficient = width / max_width
        new_width = int(max_width)
        new_height = int(height / coefficient)
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return image


def cropCardFromPicture(contour_of_card, image):
    # CONTOUR OF CARD AS RECTANGLE SHAPE:
    pts = contour_of_card.reshape(4,2)
    rect = np.zeros((4,2), dtype = "float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]         # top left
    rect[2] = pts[np.argmax(s)]         # bottom right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]      # top right
    rect[3] = pts[np.argmax(diff)]      # bottom left

    (tl, tr, br, bl) = rect             # create figure (not rectangle yet)
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    minWidth = min(int(widthA), int(widthB))

    dst = np.array([                    # destination points
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # ROTATE IF NEEDED:
    if minWidth > maxHeight:
        warp = ndimage.rotate(warp.copy(), 90)

    # SCALE:
    scaled = cv2.resize(warp, (computing.card_width, computing.card_height), interpolation = cv2.INTER_CUBIC)

    # SHOW THE OUTPUT:
    cv2.imshow('Output', scaled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return scaled


def findCardShape(colour_image, filename):
    gray_image = cv2.cvtColor(colour_image, cv2.COLOR_BGR2GRAY)
    # EDGES:
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
            cards.append(approx)

    # SAVING:
    # computing.save_file('e' + filename, 'testing_output/', edge_dilation)
    cv2.drawContours(colour_image, cards, -1, (0, 255, 0), 3)
    computing.save_file(filename, '1_find_shape_of_card/', colour_image)
    # print(filename, len(cnts), len(cards))

    return cards


def findRibbon(image, file_name):
    width, height = image.shape[:2]
    area = width*height
    results_array, masks_array = computing.find_colour_count(image.copy(), file_name)
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
            # if cv2.contourArea(c) > 3000 and cv2.arcLength(c, closed=True) > 200:     #(ref 189x111 = 20979)
            if cv2.contourArea(c) > int(area/8):
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
    # TODO binary tree
    print('CARD: ', card.name_of_file)
    print('RIBBON: ', card.isRibbon)


def read_pictures():
    types = ('*.jpg', '*.JPG')
    files_list = []
    for type in types:
        files_list.extend(sorted(glob.glob(computing.test_input + type)))
    for n, file in enumerate(files_list):
        # if n == 2:
        #  	break
        picture_processing(file)
    print('TESTING pictures count:', len(files_list))


if __name__ == '__main__':
    # REFERENCE PICTURES:
    computing.read_information(computing.reference_input)
    computing.compute_parameters()
    #-- for testing--
    for file_key in sorted(computing.facts_dictionary.keys()):
        # READ FILE:
        filename = os.path.join(os.getcwd(), computing.reference_input + file_key)
        original_image = cv2.imread(filename, cv2.IMREAD_COLOR)
        findRibbon(original_image, 'A' + file_key)

    # TESTED PICTURES:
    read_pictures()

