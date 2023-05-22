#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import csv

#from image import Image
from data import annotations_file
from transformation import Transformation
from camera import Camera


class Annotations():

    def __init__(self, image, Camera: Camera, filename: str, distorted: bool):

        self.distorted = distorted
        self.Camera = Camera
        self.pixel_sequences = self.__read_pixel_sequences_from_csv__(filename)                
        self.splines = self.__interpolate_pixel_sequences__(self.pixel_sequences)


    def __read_pixel_sequences_from_csv__(self, filename):

        pixel_sequences: list[list[np.ndarray]] = []

        with open(annotations_file) as csv_file:
            for row in csv.reader(csv_file):
                if row[0] == str(filename + '.jpg'):
                    shape = row[5]

                    idx_x = shape.find("all_points_x") + 15
                    idx_y = shape.find("all_points_y") + 15
                    x_list = [int(x) for x in shape[idx_x:idx_y-18].split(",")]
                    y_list = [int(y) for y in shape[idx_y:-2].split(",")]

                    pixel_sequence: list[np.ndarray] = []
                    for x, y in zip(x_list, y_list):
                        pixel = np.array([[x], [y]])
                        pixel_sequence.append(pixel)

                    if self.distorted == True:
                        pixel_sequence = Camera.undistort_points(self.Camera, pixel_sequence)

                    pixel_sequences.append(pixel_sequence)
        return pixel_sequences


    def __interpolate_pixel_sequences__(self, pixel_sequences: list[list[np.ndarray]]):
        splines: list[list[np.ndarray]] = []
        for pixel_sequence in pixel_sequences:
            spline = Transformation.interpolate_spline_linspace(pixel_sequence, 10, smoothing=0.5)
            splines.append(spline)
        return splines
