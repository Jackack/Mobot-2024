import time

import numpy as np
import cv2

from linedetect import *
from drive import *
detector = LineDetection()

from picamera2 import Picamera2


def transformROI(img):
    pts1 = np.float32([[0, 260], [640, 260],
                       [0, 400], [640, 400]])
    pts2 = np.float32([[0, 0], [400, 0],
                       [0, 640], [400, 640]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(frame, matrix, (500, 600))

p = 2
drv = Drive()
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280,720)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

def main():
    drv.forward()

    while True:
        frame = picam2.capture_array()

        small_to_large_image_size_ratio = 0.2
        frame = cv2.resize(frame, # original image
                    (0,0), # set fx and fy, not the final size
                    fx=small_to_large_image_size_ratio,
                    fy=small_to_large_image_size_ratio,
                    interpolation=cv2.INTER_NEAREST)

        # frame = transformROI(cam.read())
        lineFrame, xs = detector.detectLine(frame)
        if xs is None:
            continue

        # viz
        cv.imshow("frame", lineFrame)
        if (cv.waitKey(1) & 0xFF == ord('q')):
            break

        # center the x-coords
        xs = xs - frame.shape[1]/2

        # goal: keep the xs' centered (0)
        xs_slice = xs[xs.shape[0] - 10 : xs.shape[0] - 1]
        err = np.mean(xs_slice)

        # proportional control
        drv.steer(-p * err)
        print(-p*err)
    vid.release()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        drv.stopcar()