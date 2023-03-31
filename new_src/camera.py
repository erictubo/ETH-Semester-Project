#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class Camera:

    def __init__(self, alpha_u, alpha_v, u0, v0):
        self.alpha_u = alpha_u
        self.alpha_v = alpha_v
        self.u0      = u0
        self.v0      = v0

        self.FoV_u = 2*np.arctan(2*self.u0 / 2*self.alpha_u)
        self.FoV_v = 2*np.arctan(2*self.v0 / 2*self.alpha_v)
        self.K     = np.array([[self.alpha_u, 0, self.u0],
                               [0, self.alpha_v, self.v0],
                               [0, 0, 1]])