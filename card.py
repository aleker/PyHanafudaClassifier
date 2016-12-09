import cv2
import numpy as np
import computing
from enum import Enum


class Colour(Enum):
    RED = 0
    BLUE = 1
    WHITE = 2


# TODO make class
class Card:
    month = None
    points = None
    isRibbon = -1
    colours_count_array = []
    huMoments = []

    def __init__(self, name_of_file):
        self.name_of_file = name_of_file
