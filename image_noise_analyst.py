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
bins = np.arange(256).reshape(256,1)


def draw_str(dst, (x, y), s):
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.CV_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)


class NoiseAnalyst():

    """
    Analyse the noise of given image.

    Usage:
            na = NoiseAnalyst(filename)
            na.show()
    """

    # Public
    img = None

    # Windows' name
    test_winname = "test"
    ctrl_panel_winname = "control_panel"
    spectrum_winname = "spectrum"
    hist_winname = "histogram"

    sel = None # ROI

    def __init__(self, filename=testPath):
        """
        Load image in gray scale.
        """
        self.img = cv2.imread(filename, False)
        print 'Image size: ' + str(self.img.shape)

        self.roiNeedUpadte = False
        self.dragStart = None

        # save images temporarily
        self.tmp = self.img

        # DFT result fro self.img
        self.dft4img = None
    # ------------------------------------- Image processing
    def get_dft(self, img2dft=None, showdft=False):
        """
        Return the spectrum in log scale.
        """
        if img2dft == None:
            img2dft = self.img
        dft_A = cv2.dft(np.float32(self.img),flags = cv2.DFT_COMPLEX_OUTPUT|cv2.DFT_SCALE)
        dft_A = np.fft.fftshift(dft_A)

        if showdft:
            self.show_specturm(dft_A)
        return dft_A

    def remove_sin_noise(self):
        """
        Warning: this function is unoptimized and may runs slowly.

        Show the origin image with 3 trackbar.
        These bars control the direction, amplitude and phase 
        of the compensation sin wave, respectively.
        A result window shows the result.
        """
        h, w = self.img.shape[:2]
        small_img = cv2.resize(self.img, (w / 2, h / 2))
        cv2.imshow(self.test_winname, small_img)
        cv2.createTrackbar("A*100", self.test_winname,
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
            if ch == 27:  # Esc
                break
        cv2.destroyWindow(self.test_winname)

    def remove_vertical_sin_noise(self):
        A = 12
        amp = 10
        pha = 0
        h, w = self.img.shape[:2]
        small_img = cv2.resize(self.img, (w / 2, h / 2))
        cv2.imshow(self.test_winname, small_img)
        cv2.imshow(self.ctrl_panel_winname, np.zeros((100,600), np.uint8))
        cv2.createTrackbar("A*100", self.ctrl_panel_winname,
                           A, 100, self.update_vertical_sine_win)
        cv2.createTrackbar("amp", self.ctrl_panel_winname,
                           amp, 255, self.update_vertical_sine_win)
        cv2.createTrackbar("pha", self.ctrl_panel_winname,
                           pha, 360, self.update_vertical_sine_win)
        self.update_vertical_sine_win()
        print "Reducing sinusoidal noise, Press a to accspt"
        while True:
            ch = cv2.waitKey()
            if ch == 27:  # Esc
                break
            if ch == 97:
                self.img = self.tmp
                break
        cv2.destroyAllWindows()

    def apply_gaussian_filter(self):
        """
        Apply gaussion low pass filter.
        """
        max_ksize = min(self.img.shape[0], self.img.shape[1])
        cv2.imshow(self.test_winname, self.img)
        cv2.imshow(self.ctrl_panel_winname, np.zeros((100, 600), np.uint8))
        cv2.createTrackbar(
            "kize=2n+1:", self.ctrl_panel_winname, 3, (max_ksize - 1) / 2, self.update_gaussian_filter_win)
        self.update_gaussian_filter_win()
        print "Reducing high frequency noise, Press a to accept"
        while True:
            ch = cv2.waitKey()
            if ch == 27:
                break
            if ch == 97:
                self.img = self.tmp
                break
        cv2.destroyAllWindows()

    def apply_butterworth_filter(self):
        """
        Apply Butterworth low pass filter.
        The code is derived from Paragraph 4.8.2, 
        Butterworth Lowpass Filters, 
        "Digital image Processing (3rd edition)" by R.C. Gonzalez.
        """
        max_ksize = max(self.img.shape[0], self.img.shape[1])
        self.dft4img = self.get_dft(self.img, showdft=True)
        cv2.imshow(self.test_winname, self.img)
        cv2.imshow(self.ctrl_panel_winname, np.zeros((100, 600), np.uint8))
        cv2.createTrackbar(
            "stopband**2", self.ctrl_panel_winname, 3, (max_ksize - 1) / 2, self.update_butterworth_win)
        self.update_butterworth_win()
        print "Reducing high frequency noise, Press a to accept"
        while True:
            ch = cv2.waitKey()
            if ch == 27:
                break
            if ch == 97:
                self.img = self.tmp
                plt.imshow(self.img)
                plt.show()
                break
        cv2.destroyAllWindows()

    def get_butterworth_filter(self, stopband2=10, order=3, showdft=False):
        """
        Get Butterworth filter in frequency domain.
        """
        h, w = self.dft4img.shape[0], self.dft4img.shape[1]
        P = h/2
        Q = w/2
        dst = np.zeros((h, w, 2), np.float64)
        for i in range(h):
            for j in range(w):
                r2 = float((i-P)**2+(j-Q)**2)
                if r2 == 0:
                    r2 = 1.0
                dst[i,j] = 1/(1+(r2/stopband2)**order)
        dst = np.float64(dst)
        if showdft:
            cv2.imshow("butterworth", cv2.magnitude(dst[:,:,0], dst[:,:,1]))
        return dst

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
                sinimg[i, j] = np.uint8(amp * (sin(A * i + B * j + pha) + 1))
        return sinimg

    def get_vertical_sin_image(self, A, amp, pha):
        """
        Get horizontal sine image for compensation.
        Faster than get_sine_img().
        """
        h, w = self.img.shape[:2]
        tmp = range(h)
        # create sine wave in 1 col
        tmp = np.array(range(h), np.float64)
        tmp = np.multiply(
            np.add(np.sin(np.add(np.multiply(tmp, A), pha)), 1), amp)
        tmp = tmp.tolist()
        sinimg = np.uint8(np.array([tmp] * w)).transpose()
        # self.show(sinimg)
        return sinimg

    def set_roi(self):
        """
        Display a image for ROI selection.
        Press a to accept this ROI and self.img will
        be OVERRIDE with the new image
        """
        cv2.imshow(self.test_winname, self.img)
        cv2.setMouseCallback(self.test_winname, self.onmouse)
        while True:
            ch = cv2.waitKey()
            if ch == 27: # Esc
                break
            elif self.roiNeedUpadte and ch == 97: # a
                print "Accept ROI (minX, minY, maxX, maxY): " +  str(self.sel)
                self.roiNeedUpadte = False
                break
        cv2.destroyAllWindows()
        self.img = self.img[self.sel[1]:self.sel[3],self.sel[0]:self.sel[2]]

    def hist_lines(self, im, showhist=False):
        """
        Calculate histogram of given image.
        plot the histogram if showhist==True.

        @return     hist_item       histogram in a 2d image.
        """
        h = np.zeros((300,256,3))
        if len(im.shape)!=2:
            print "hist_lines applicable only for grayscale images"
            #print "so converting image to grayscale for representation"
            im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        hist_item = cv2.calcHist([im],[0],None,[256],[0,256])
        cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
        hist=np.int32(np.around(hist_item))
        for x,y in enumerate(hist):
            cv2.line(h,(x,0),(x,y),(255,255,255))
        y = np.flipud(h)
        # y = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
        if showhist:
            cv2.imshow(self.hist_winname, y)
        return y

    # ---------------------------------------- User interface
    def show(self, img2show):
        """
        Show a image.
        """
        cv2.imshow(self.test_winname, img2show)
        while True:
            ch = cv2.waitKey()
            if ch == 27:  # Esc
                break
        cv2.destroyAllWindows()

    def update_vertical_sine_win(self, dummy=None):
        """
        Update output image with trackbar.
        """
        A = cv2.getTrackbarPos("A*100",
                               self.ctrl_panel_winname)
        amplitude = cv2.getTrackbarPos("amp",
                                       self.ctrl_panel_winname)
        phase = cv2.getTrackbarPos("pha",
                                   self.ctrl_panel_winname)
        sin_img = self.get_vertical_sin_image(
            A * 0.01, amplitude, phase * 0.017)
        dst = np.subtract(self.img, sin_img)
        self.tmp = dst
        self.hist_lines(dst, showhist=True)
        mi, ma, miloc, maloc = cv2.minMaxLoc(dst)
        me, mestd = cv2.meanStdDev(dst)
        s1 = '%d~%d' % (mi, ma)
        s2 = '%.2f +- %.2f' % (me, mestd)
        draw_str(dst, (10,20), s1)
        draw_str(dst, (10,40), s2)
        cv2.imshow(self.test_winname, dst)

    def update_sine_win(self, dummy=None):
        """
        Update output image with trackbar.
        The direction is user defined.

        TODO: speed up self.get_sine_img()
        """
        pass

    def update_gaussian_filter_win(self, dummy=None):
        """
        Update Gaussian Kernel param and the result image.
        """
        ks = cv2.getTrackbarPos("kize=2n+1:",
                                self.ctrl_panel_winname)
        kernel = cv2.getGaussianKernel(ks*2+1, 0)
        dst = cv2.filter2D(self.img, -1, kernel)
        self.tmp = dst
        self.get_dft()
        cv2.imshow(self.test_winname, dst)

    def update_butterworth_win(self, dummy=None):
        """
        Update Butterworth filter param and the result image.
        """
        sb = cv2.getTrackbarPos("stopband**2",
                                self.ctrl_panel_winname)
        if sb == 0:
            sb = 1
            print "Stopband should be more than 0. Reset to 1."
        bw_filter = self.get_butterworth_filter(stopband2=sb, showdft=True)
        dst_complex = bw_filter * self.dft4img#cv2.multiply(self.dft4img, bw_filter)
        dst_complex = cv2.idft(np.fft.ifftshift(dst_complex))
        dst = np.uint8(cv2.magnitude(dst_complex[:,:,0], dst_complex[:,:,1]))
        self.tmp = dst
        self.get_dft(self.tmp, showdft=True)
        self.hist_lines(dst, showhist=True)
        cv2.imshow(self.test_winname, dst)

    def show_specturm(self, dft_result):
        """
        Show spectrun graph.
        """
        # Split fourier into real and imaginary parts
        image_Re, image_Im = cv2.split(dft_result)

        # Compute the magnitude of the spectrum Mag = sqrt(Re^2 + Im^2)
        magnitude = cv2.sqrt(image_Re ** 2.0 + image_Im ** 2.0)

        # Compute log(1 + Mag)
        log_spectrum = cv2.log(1.0 + magnitude)

        # Rearrange the quadrants of Fourier image so that the origin is at
        # the image center
        # shift_dft(log_spectrum, log_spectrum)

        # normalize and display the results as rgb
        cv2.normalize(log_spectrum, log_spectrum, 0.0, 1.0, cv2.cv.CV_MINMAX)
        # plt.imshow(log_spectrum)
        # plt.show()
        cv2.imshow(self.spectrum_winname, log_spectrum)

    def plot_vertically(self, xPos=1):
        """
        Plot pixel in a line (x == xPos), which helps to 
        find out the period of sinusoidal noise.
        """
        l = na.img[xPos, :]
        plt.plot(range(len(l)), l)
        plt.show()

    def onmouse(self, event, x, y, flags, param):
        """
        Mouse callback when mouse event detected in the window.
        Note: This function is only used for ROI setting.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragStart = x, y
            self.sel = 0,0,0,0
        elif self.dragStart:
            #print flags
            if flags & cv2.EVENT_FLAG_LBUTTON:
                minpos = min(self.dragStart[0], x), min(self.dragStart[1], y)
                maxpos = max(self.dragStart[0], x), max(self.dragStart[1], y)
                self.sel = minpos[0], minpos[1], maxpos[0], maxpos[1]
                img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(img, (self.sel[0], self.sel[1]), (self.sel[2], self.sel[3]), (0,255,255), 1)
                cv2.imshow(self.test_winname, img)
            else:
                patch = self.img[self.sel[1]:self.sel[3], self.sel[0]:self.sel[2]]
                self.hist_lines(patch, showhist=True)
                cv2.destroyWindow("patch")
                cv2.imshow("patch", patch)
                self.get_dft(img2dft=patch, showdft=True)
                print "Press a to accept the ROI"
                self.roiNeedUpadte = True
                self.dragStart = None

if __name__ == "__main__":
    na = NoiseAnalyst()

    # ROI test
    # na.set_roi()

    # Butterworth filter test
    # Recommand to test ROI first
    fshift = na.get_dft()
    na.apply_butterworth_filter()

    # Gaussian filter test
    # Recommand to test without butterworth test
    # na.apply_gaussian_filter()

    # Remove Vertical sine wave noise
    # Note: Butterworth filter performs better in our case
    # na.remove_vertical_sin_noise

