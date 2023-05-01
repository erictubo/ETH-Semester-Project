#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from data import track_width
from features import Features


class Image:
    """
    Data Object\\
    Initialised automatically by KeyFrame object\\
    Use keyframe.Image to access data
    """

    def __init__(self, image):
        self.image = image

        # Extracted from image
        self.image_height = self.image.shape[0]
        self.image_width = self.image.shape[1]
        self.v0 = self.image_height/2
        self.u0 = self.image_width/2

        # Extracted image features
        # self.vanishing_point = Features.get_vanishing_point()
        #self.track_width_pixels = Features.get_track_width_pixels()
        # self.u_track_middle = Features.get_track_middle()



