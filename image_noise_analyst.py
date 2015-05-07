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
import numpy as np
from math import *
import matplotlib.pylab as plt

testPath = './test.bmp'

def shift_dft(src, dst=None):
    '''
        Rearrange the quadrants of Fourier image so that the origin is at
        the image center. Swaps quadrant 1 with 3, and 2 with 4.

        src and dst arrays must be equal size & type
    '''

    if dst is None:
        dst = np.empty(src.shape, src.dtype)
    elif src.shape != dst.shape:
        raise ValueError("src and dst must have equal sizes")
    elif src.dtype != dst.dtype:
        raise TypeError("src and dst must have equal types")

    if src is dst:
        ret = np.empty(src.shape, src.dtype)
    else:
        ret = dst

    h, w = src.shape[:2]

    cx1 = cx2 = w/2
    cy1 = cy2 = h/2

    # if the size is odd, then adjust the bottom/right quadrants
    if w % 2 != 0:
        cx2 += 1
    if h % 2 != 0:
        cy2 += 1

    # swap quadrants

    # swap q1 and q3
    ret[h-cy1:, w-cx1:] = src[0:cy1 , 0:cx1 ]   # q1 -> q3
    ret[0:cy2 , 0:cx2 ] = src[h-cy2:, w-cx2:]   # q3 -> q1

    # swap q2 and q4
    ret[0:cy2 , w-cx2:] = src[h-cy2:, 0:cx2 ]   # q2 -> q4
    ret[h-cy1:, 0:cx1 ] = src[0:cy1 , w-cx1:]   # q4 -> q2

    if src is dst:
        dst[:,:] = ret

    return dst

class NoiseAnalyst():
    """
    Analyse the noise of given image.

    Usage:
            na = NoiseAnalyst(filename)
            na.show()
    """

    # Public
    img = None 
    test_winname = "test"

    def __init__(self, filename=testPath):
        """
        Load image in gray scale.
        """
        self.img = cv2.imread(filename, False)

    # ------------------------------------------- Processing
    def showDFT(self):
        """
        Return the spectrum in log scale.
        """
        h, w = self.img.shape[:2]
        realInput = self.img.astype(np.float64)
        # perform an optimally sized dft
        dft_M = cv2.getOptimalDFTSize(w)
        dft_N = cv2.getOptimalDFTSize(h)
        # copy A to dft_A and pad dft_A with zeros
        dft_A = np.zeros((dft_N, dft_M, 2), dtype=np.float64)
        dft_A[:h, :w, 0] = realInput
        # no need to pad bottom part of dft_A with zeros because of
        # use of nonzeroRows parameter in cv2.dft()
        cv2.dft(dft_A, dst=dft_A, nonzeroRows=h)

        
        return dft_A

    def show_sin_wave(self):
        """
        Show the origin image with 3 trackbar.
        These bars control the direction, amplitude and phase 
        of the compensation sin wave, respectively.
        A result window shows the result.
        """
        h, w = self.img.shape[:2]
        small_img = cv2.resize(self.img,(w/2,h/2))
        cv2.imshow(self.test_winname, small_img)
        cv2.createTrackbar("A", self.test_winname, 
            0, 100, self.update_sine_win)
        cv2.createTrackbar("B", self.test_winname, 
            1, 100, self.update_sine_win)
        cv2.createTrackbar("amp", self.test_winname, 
            0, 255, self.update_sine_win)
        cv2.createTrackbar("pha", self.test_winname, 
            0, 360, self.update_sine_win)
        self.update_sine_win()
        while True:
            ch = cv2.waitKey() 
            if ch == 27: # Esc
                break
        cv2.destroyWindow(self.test_winname)

    def get_sine_img(self, A, B, amp, pha):
        """
        create sine image for compensation.
        TODO: speed up.
        """
        # P(x,y) = Amp * sin( A*x + B*y + Pha)
        h, w = self.img.shape[:2]
        sinimg = np.zeros((h, w, 1), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                sinimg[i,j] = np.uint8(amp * (sin(A*i+B*j+pha)+1))
        return sinimg

    def get_horizontal_sin_img(self, A, amp, pha):
        """
        Get horizontal sine image for compensation.
        Faster than get_sine_img().
        """
        h, w = self.img.shape[:2]
        tmp = range(h)
        # create sine wave in 1 col
        tmp = np.array(range(h), np.float64) 
        tmp = np.multiply(np.add(np.sin(np.add(np.multiply(tmp,A),pha)),1),amp)
        tmp = np.uint8(tmp).tolist()
        # copy to other cols
        sinimg = np.array([tmp]*w)
        return sinimg
        


    # ------------------------------------------- Highgui 
    def show(self, img2show):
        """
        Show a image.
        """
        cv2.imshow(self.test_winname, img2show)
        while True:
            ch = cv2.waitKey() 
            if ch == 27: # Esc
                break
        cv2.destroyWindow(self.test_winname)

    def update_sine_win(self, dummy=None):
        """
        Update output image with trackbar.
        """
        A = cv2.getTrackbarPos("A", 
            self.test_winname)
        B = cv2.getTrackbarPos("B",
            self.test_winname)
        amplitude = cv2.getTrackbarPos("amp",
            self.test_winname)
        phase = cv2.getTrackbarPos("pha",
            self.test_winname)

    def show_specturm(self, dft_result):
        # Split fourier into real and imaginary parts
        image_Re, image_Im = cv2.split(dft_result)

        # Compute the magnitude of the spectrum Mag = sqrt(Re^2 + Im^2)
        magnitude = cv2.sqrt(image_Re**2.0 + image_Im**2.0)

        # Compute log(1 + Mag)
        log_spectrum = cv2.log(1.0 + magnitude)

        # Rearrange the quadrants of Fourier image so that the origin is at
        # the image center
        shift_dft(log_spectrum, log_spectrum)

        # normalize and display the results as rgb
        cv2.normalize(log_spectrum, log_spectrum, 0.0, 1.0, cv2.cv.CV_MINMAX)
        plt.imshow(log_spectrum)
        plt.show()
      


if __name__ == "__main__":
    na = NoiseAnalyst()
    # na.show(img2show=na.img)
    na.showDFT()
    # na.show_sin_wave()
    # na.get_sine_img(0.03,0,10,10)
    # na.get_horizontal_sin_img(0.03,100,10)

