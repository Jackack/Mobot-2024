import time

import numpy as np
import cv2 as cv

class LineDetection:
    def __init__(self) -> None:
        self.polynomial_1 = (0, 0, 200)

    def threshold(self, img):
        threshold = np.percentile(img[:, :, 1], 85)
        img = np.where((img[:, :, 1] > threshold * np.max(img))[:, :, None], img, np.zeros(img.shape))
        return img

    def polymask(self, img, xs, width):
        print("img shape", img.shape)
        coords = np.mgrid[:img.shape[0], :img.shape[1]]
        xs = xs[:, None]
        print("xs shape", xs.shape)
        xdist = np.abs(coords[1, :, :] - np.repeat(xs, img.shape[1], axis=1))
        return np.where((xdist > width)[:, :, None], img, np.zeros(img.shape))

    def polyband_yx(self, line_y, line_x, poly, width, include):
        # return Nx2 array, where col 1 = line_y, col 2 = line x
        # if include is set, (y, x) points inside the polynomial band
        # of width width is returned. Else, points outside the band
        # is returned.

        yx = np.int16(np.hstack((line_y[:, None], line_x[:, None])))
        print(yx.shape)

        xs = np.polyval(poly, yx[:, 0])
        if include:
            yx = yx[np.abs(yx[:, 1] - xs) < width, :]
        else:
            yx = yx[np.abs(yx[:, 1] - xs) >= width, :]
        return yx


    def polyfit(self, img, poly, width, include):
        line_y, line_x = np.nonzero(np.linalg.norm(img, axis=2))

        yx = self.polyband_yx(line_y, line_x, poly, width, include)

        if yx.shape[0] == 0:
            return None, None, None

        # polyfit
        p, residual, rank, singular_values, rcond = np.polyfit(yx[:, 0], yx[:, 1], 3, full=True)
        ys = np.linspace(0, img.shape[0] - 1, img.shape[0])
        xs = np.polyval(p, ys)

        return xs, ys, p


    def detectLine(self, img_in):
        # convert image to HLS, and threshold by L
        img = np.float32(cv.cvtColor(img_in, cv.COLOR_BGR2HLS)) / 255
        # dilate

        # zero out all pixels outside of a band centered at the last estimated polynomial
        ys = np.linspace(0, img.shape[0] - 1, img.shape[0])
        xs = np.polyval(self.polynomial_1, ys)
        thres_img = self.threshold(img)

        # kernel = np.ones((3, 3))
        # thres_img = cv.erode(thres_img, kernel, iterations=1)
        # thres_img = cv.dilate(thres_img, kernel, iterations=1)

        xs1, ys, p1 = self.polyfit(thres_img, self.polynomial_1, 100, True)
        if not p1 is None:
            self.polynomial_1 = p1

        out_img = img_in

        if xs1 is None:
            return out_img

        # fit a second polynomial
        xs2, ys, p2 = self.polyfit(thres_img, p1, 15, False)

        if xs2 is None:
            return out_img

        yx1 = np.int16(np.hstack((ys[:, None], xs[:, None])))
        yx1 = yx1[yx1[:, 1] < img.shape[1]]
        yx1 = yx1[yx1[:, 1] >= 0]

        out_img[yx1[:, 0], yx1[:, 1], :] = np.array([0, 0, 255])
        out_img[yx1[:, 0], np.clip(yx1[:, 1] + 1, 0, out_img.shape[1]) - 1, :] = np.array([0, 0, 255])
        out_img[yx1[:, 0], np.clip(yx1[:, 1] - 1, 0, out_img.shape[1]) - 1, :] = np.array([0, 0, 255])

        # xs is the list of x-coordinates in the image

        return out_img, xs


if __name__ == "__main__":
    # define a video capture object
    vid = cv.VideoCapture(1)
    detector = LineDetection()
    while True:
        ret, frame = vid.read()
        print(ret)
        if ret:
            small_to_large_image_size_ratio = 0.2
            small_img = cv.resize(frame, # original image
                       (0,0), # set fx and fy, not the final size
                       fx=small_to_large_image_size_ratio,
                       fy=small_to_large_image_size_ratio,
                       interpolation=cv.INTER_NEAREST)
            print(small_img.shape)

            cv.imshow("frame", detector.detectLine(small_img))

            if (cv.waitKey(1) & 0xFF == ord('q')):
                break




    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv.destroyAllWindows()