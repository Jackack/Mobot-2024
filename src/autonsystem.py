import time

import numpy as np
import cv2

from linedetect import *
# from camera import *
# from drive import *


detector = LineDetection()
# cam = Camera()
# drive = Drive()


def transformROI(img):
    pts1 = np.float32([[0, 260], [640, 260],
                       [0, 400], [640, 400]])
    pts2 = np.float32([[0, 0], [400, 0],
                       [0, 640], [400, 640]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(frame, matrix, (500, 600))

p = 0.001
vid = cv.VideoCapture(0)

while True:
    # read frame
    ret, frame = vid.read()

    small_to_large_image_size_ratio = 0.2
    frame = cv2.resize(frame, # original image
                (0,0), # set fx and fy, not the final size
                fx=small_to_large_image_size_ratio,
                fy=small_to_large_image_size_ratio,
                interpolation=cv2.INTER_NEAREST)

    if not ret:
        time.sleep(0.2)
        continue

    # frame = transformROI(cam.read())
    lineFrame, xs = detector.detectLine(frame)

    # viz
    cv.imshow("frame", lineFrame)
    if (cv.waitKey(1) & 0xFF == ord('q')):
        break

    # center the x-coords
    xs = xs - frame.shape[1]/2

    # goal: keep the xs' centered (0)
    err = np.sum(xs)

    # proportional control
    # drive.setSteer(p * err)
    print(p*err)

