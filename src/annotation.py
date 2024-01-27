#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Handles the Annotation class for processing manual image annotations of railway tracks.

Loads annotations from CSV files, processes them into 2D splines, and provides visualization utilities. Used as part of Keyframe objects.
"""

import numpy as np
import csv
import cv2

#from image import Image
from data import path_to_annotations_0, path_to_annotations_1
from transformation import Transformation
from camera import Camera
from visualization import Visualization


class Annotation():
    """
    Represents manual annotation of railway tracks in an image.
    Loads, processes, and provides access to annotated pixel sequences and their interpolated splines.
    """

    def __init__(self, image, camera: Camera, filename: str, distorted: bool, int_spacing=8, int_smoothing=0.5):
        """
        Initialize an Annotation object for a given image and camera.
        Args:
            image: The image to annotate.
            camera: Camera object (for undistortion).
            filename: Frame identifier (string or int).
            distorted: Whether to undistort annotation points.
            int_spacing: Interpolation spacing (default: 8).
            int_smoothing: Interpolation smoothing (default: 0.5).
        """
        self.image = image
        self.camera = camera
        self.filename = filename
        self.distorted = distorted

        self.pixel_sequences = self.__read_pixel_sequences_from_csv__(filename)                
        self.splines = self.__interpolate_pixel_sequences__(self.pixel_sequences, int_spacing, int_smoothing)
        self.array = self.__convert_to_single_array__(self.splines)


    # Hidden methods: called at initialisation

    def __read_pixel_sequences_from_csv__(self, filename):
        """
        Read pixel sequences for the given frame from the annotation CSV file.
        Args:
            filename: Frame identifier (string or int).
        Returns:
            List of pixel sequences (list of list of np.ndarray).
        """
        pixel_sequences: list[list[np.ndarray]] = []

        if self.camera.id == 0:
            path_to_annotations = path_to_annotations_0
        elif self.camera.id == 1:
            path_to_annotations = path_to_annotations_1
        annotations_file = path_to_annotations

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
                        pixel_sequence = self.camera.undistort_points(pixel_sequence)

                    pixel_sequences.append(pixel_sequence)
        return pixel_sequences


    def __interpolate_pixel_sequences__(self, pixel_sequences: list[list[np.ndarray]], spacing, smoothing):
        """
        Interpolate each pixel sequence to increase point density and smoothness.
        Args:
            pixel_sequences: List of pixel sequences.
            spacing: Desired spacing between points.
            smoothing: Smoothing factor.
        Returns:
            List of interpolated splines (list of list of np.ndarray).
        """
        splines: list[list[np.ndarray]] = []
        for pixel_sequence in pixel_sequences:
            spline = Transformation.interpolate_spline(points=pixel_sequence, desired_spacing=spacing, smoothing=smoothing)
            for pixel in spline:
                pixel[0] = np.rint(pixel[0])
                pixel[1] = np.rint(pixel[1])
            splines.append(spline)
        return splines
    

    def __convert_to_single_array__(self, splines: list[list[np.ndarray]]):
        """
        Converts a list of splines to a single numpy array.
        Args:
            splines: List of splines (list of list of np.ndarray).
        Returns:
            Numpy array of all points (N, 2).
        """
        array = np.empty((0,2))
        for spline in splines:
            array_track = Transformation.convert_points_list(spline, to_type="array")
            array = np.append(array, array_track, axis=0)
        return array


    # Public methods

    def visualize_splines(self, visual=None, color: tuple=(255,0,0)):
        """
        Draw the interpolated splines on the image.
        Args:
            visual: Optional image to draw on (default: copy of self.image).
            color: Color for the splines (default: red).
        Returns:
            Image with splines drawn.
        """
        if visual is None:
            visual = self.image.copy()

        for spline in self.splines:
            Visualization.draw_on_image(visual, spline, False, color)
        return visual


    def visualize_points(self, visual=None, color: tuple=(255,255,0)):
        """
        Draw the original annotated points on the image.
        Args:
            visual: Optional image to draw on (default: copy of self.image).
            color: Color for the points (default: yellow).
        Returns:
            Image with points drawn.
        """
        if visual is None:
            visual = self.image.copy()

        for pixel_sequence in self.pixel_sequences:
            Visualization.draw_on_image(visual, pixel_sequence, False, color)
        return visual
    
    
    def visualize_splines_and_points(self, visual = None, colors: list[tuple]=[(255,0,0), (255,255,0)]):
        """
        Draw both splines and original points on the image.
        Args:
            visual: Optional image to draw on.
            colors: List of two colors for splines and points.
        Returns:
            Image with both splines and points drawn.
        """
        visual = self.visualize_splines(visual, colors[0])
        visual = self.visualize_points(visual, colors[1])
        return visual