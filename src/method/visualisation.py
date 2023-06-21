#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# External libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Data & methods
from transformation import Transformation


class Visualisation:

    @staticmethod
    def draw_on_image(image: cv2.Mat, pixels: list[np.ndarray[int]], lines: list[np.ndarray[int]], colors_BGR = (0,0,255), thickness=3, outside_pixels="ignore") -> cv2.Mat:
        
        h = image.shape[0]
        w = image.shape[1]

        if pixels:
            pixels_in_frame = 0

            for i, pixel in enumerate(pixels):
                assert pixel.shape in [(2,), (2,1)], pixel.shape
                u = int(pixel[0])
                v = int(pixel[1])

                if outside_pixels == "ignore":
                    if not (0 <= u <= w) or not (0 <= v <= h):
                        continue
                        #print("ignored pixel [u,v] =", str([u,v]), "since not within image boundaries: [h,w] =", str([h,w]))

                elif outside_pixels == "assert":
                    assert (0 <= u <= w) and (0 <= v <= h), print("pixel [u,v] =", str([u,v]), "not within image boundaries: [h,w] =", str([h,w]))

                if type(colors_BGR) == list: color = colors_BGR[i]
                else: color = colors_BGR

                pixels_in_frame += 1

                cv2.circle(image, (u, v), 3, color, -1)

            # print(pixels_in_frame, "/", len(pixels), "pixels inside image")

        if lines:
            for line in lines:
                assert line.shape == (2,2)
                u1 = int(line[0,0])
                v1 = int(line[1,0])
                u2 = int(line[0,1])
                v2 = int(line[1,1])

                if type(colors_BGR) == list: color = colors_BGR[i]
                else: color = colors_BGR

                cv2.line(image, (u1,v1), (u2,v2), color, thickness)

        return image


    @staticmethod
    def convert_consecutive_points_to_3D_lines(points: list[np.ndarray[int]]) -> list[np.ndarray[int]]:
        lines = []
        for i in range(len(points)-1):
            point = points[i]
            assert point.shape[0] == 3, point.shape
            start_point = points[i]
            end_point = points[i+1]
            line = np.c_[start_point, end_point]
            assert line.shape == (3,2), line.shape
            lines.append(line)
        assert len(lines) == len(points)-1
        return lines


    @staticmethod
    def convert_consecutive_pixels_to_2D_lines(pixels: list[np.ndarray[int]]) -> list[np.ndarray[int]]:
        lines = []
        if len(pixels) >= 2:
            for i in range(len(pixels)-1):
                pixel = pixels[i]
                assert pixel.shape[0] == 2, pixel.shape
                start_pixel = pixels[i]
                end_pixel = pixels[i+1]
                line = np.c_[start_pixel, end_pixel]
                assert line.shape == (2,2), line.shape
                lines.append(line)
            assert len(lines) == len(pixels)-1
        return lines

    """
    2D Plots
    """
    @staticmethod
    def create_2D_plot(title: str):
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(projection='2d')
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        return ax
    
    @staticmethod
    def plot_XY(X, Y, color: str='blue', scale = 'equal'):
        plt.axis('equal')
        plt.scatter(X, Y, c=color)


    """
    3D Plots
    """

    @staticmethod
    def create_3D_plot(title: str):
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(projection='3d')
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_aspect('equal')
        return ax

    @staticmethod
    def plot_3D_points(ax, points: list[np.ndarray], color: str='blue', scale = 'equal'):
        X, Y, Z = Transformation.convert_points_list(points, to_type="components")
        Visualisation.plot_XYZ(ax, X, Y, Z, color, scale)

    @staticmethod
    def plot_XYZ(ax, X, Y, Z, color: str='blue', scale = 'equal'):
        ax.scatter(X, Y, Z, c=color)

        if scale == 'equal':
            # Create cubic bounding box to simulate equal aspect ratio
            max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
            Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
            Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
            Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
            for xb, yb, zb in zip(Xb, Yb, Zb):
                ax.plot([xb], [yb], [zb], 'w')

    @staticmethod
    def plot_3D_lines(ax, lines: list[np.ndarray]):
        for line in lines:
                ax.plot(line[0,:], line[1,:], line[2,:])

    @staticmethod
    def show_plot():
        plt.show()
