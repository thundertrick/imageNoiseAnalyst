#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2015-2016, Xuyang Hu <xuyanghu@yahoo.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License version 2.1
# as published by the Free Software Foundation

"""
image_noise_analyst is used to process images received from UI.

This file can be test standalone using cmd:
    python image_noise_analyst.py
"""

import cv2

testPath = './test.bmp'

class NoiseAnalyst():
    """
    Analyse the noise of given image.

    Usage:
            na = NoiseAnalyst(filename)
            na.show()
    """

    # Public
    img = None 

    def __init__(self, filename=testPath, isRGB=False):
        """
        Load image in gray scale by default. (isRGB=False)
        """
        self.img = cv2.imread(filename, not(isRGB))

    def show(self, img2show=None, winname="test"):
        if not(img2show):
            img2show = self.img
        cv2.imshow(winname, img2show)
        while True:
            ch = cv2.waitKey() 
            if ch == 27: # Esc
                break
        cv2.destroyWindow(winname)

if __name__ == "__main__":
    na = NoiseAnalyst()
    na.show()

