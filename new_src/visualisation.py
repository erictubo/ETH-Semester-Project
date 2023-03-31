#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2


class Visualisation:

    @staticmethod
    def draw_pixels_on_image(image, pixels: list[np.ndarray[int]], colors_BGR = (0,0,255)):
        
        height = image.shape[1]
        width = image.shape[0]

        for i, pixel in enumerate(pixels):
            assert pixel.shape == (2,1)
            u = pixel[0]
            v = pixel[1]
            assert 0 <= pixel[0] <= width
            assert 0 <= pixel[1] <= height

            if type(colors_BGR) == list: color = colors_BGR[i]
            else: color = colors_BGR

            cv2.circle(image, (u, v), 5, color, -1)
        return image

    @staticmethod
    def draw_lines_on_image(image, lines: list[np.ndarray], colors_BGR=(0,0,255), thickness=5):
        """
        Input: image, list of line coordinates [[u1,u2],[v1,v2]], list of colors
        """
        for line in lines:
            assert line.shape == (2,2)
            u1 = line[0,0]
            v1 = line[1,0]
            u2 = line[0,1]
            v2 = line[1,1]

            if type(colors_BGR) == list: color = colors_BGR[i]
            else: color = colors_BGR

            cv2.line(image, (u1,v1), (u2,v2), color, thickness)
        return image
    