import numpy as np
import cv2

from linedetect import *
from camera import *
from drive import *

cam = Camera()
detector = LineDetection()
drive = Drive()


def transformROI(img):
    pts1 = np.float32([[0, 260], [640, 260],
                       [0, 400], [640, 400]])
    pts2 = np.float32([[0, 0], [400, 0],
                       [0, 640], [400, 640]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(frame, matrix, (500, 600))

p = 0.1

while True:
    # read frame
    frame = transformROI(cam.read())
    lineFrame, xs = detector.detectLine(frame)

    # center the x-coords
    xs = xs - frame.shape[1]/2

    # goal: keep the xs' centered (0)
    err = np.sum(xs)

    # proportional control
    drive.setSteer(p * err)

