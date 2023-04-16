#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cv2


class Visualisation:

    @staticmethod
    def draw_pixels_on_image(image: cv2.Mat, pixels: list[np.ndarray[int]], colors_BGR = (0,0,255), outside_pixels="ignore") -> cv2.Mat:
        
        h = image.shape[0]
        w = image.shape[1]

        for i, pixel in enumerate(pixels):
            assert pixel.shape[0] == 2, pixel.shape
            u = int(pixel[0])
            v = int(pixel[1])

            if outside_pixels == "ignore":
                if not (0 <= pixel[0] <= w) or not (0 <= pixel[1] <= h):
                    print("ignored pixel [u,v] =", str([u,v]), "since not within image boundaries: [h,w] =", str([h,w]))

            elif outside_pixels == "assert":
                assert (0 <= pixel[0] <= w) and (0 <= pixel[1] <= h), print("pixel [u,v] =", str([u,v]), "not within image boundaries: [h,w] =", str([h,w]))

            if type(colors_BGR) == list: color = colors_BGR[i]
            else: color = colors_BGR

            cv2.circle(image, (u, v), 5, color, -1)
        return image
    

    @staticmethod
    def convert_consecutive_pixels_to_lines(pixels: list[np.ndarray[int]]) -> list[np.ndarray[int]]:
        lines = []
        for i in range(len(pixels)-1):
            start_pixel = pixels[i]
            end_pixel = pixels[i+1]
            line = np.c_[start_pixel, end_pixel]
            assert line.shape == (2,2), line.shape
            lines.append(line)
        assert len(lines) == len(pixels)-1
        return lines


    @staticmethod
    def draw_lines_on_image(image: cv2.Mat, lines: list[np.ndarray], colors_BGR=(0,0,255), thickness=5) -> cv2.Mat:
        """
        Input: image, list of line coordinates [[u1,u2],[v1,v2]], list of colors
        """
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
    def show_3D_plot_of_points(points: list[np.ndarray], title: str, scale: str='equal', color: str='blue'):

        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(projection='3d')
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_aspect('equal')

        X = np.zeros(len(points))
        Y = np.zeros(len(points))
        Z = np.zeros(len(points))

        for i, point in enumerate(points):
            X[i], Y[i], Z[i] = point[0], point[1], point[2]
            
        print(X)

        ax.scatter(X, Y, Z, c=color)

        if scale == 'equal':
            # Create cubic bounding box to simulate equal aspect ratio
            max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
            Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
            Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
            Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
            for xb, yb, zb in zip(Xb, Yb, Zb):
                ax.plot([xb], [yb], [zb], 'w')

        plt.show()




