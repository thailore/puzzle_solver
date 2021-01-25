#!/usr/bin/env python

import cv2
import numpy as np


def main():

    image = cv2.imread("pieces4.jpg")
    contours = get_contour_lines("pieces4.jpg")


    points = []
    for i in range(len(contours)):
        cv2.drawContours(image, contours, i, (0,255,0), 3)
        cv2.imshow('Contours', image)
        cv2.waitKey(0)
        points.append(cv2.arcLength(contours[i],True))

    canvas = np.zeros(image.shape, np.uint8)
    cv2.drawContours(canvas, contours, -1, (0, 0, 255), 3)
    cv2.imshow('Points', canvas)
    cv2.waitKey(0)

    print(len(points))

    print("Area: ", sum(cv2.contourArea(i) for i in contours))

    cv2.destroyAllWindows()


def get_contour_lines(image):
    image = cv2.imread(image)

    kernel = np.ones((5,5), np.uint8)

    dilated_image = cv2.dilate(image, kernel, iterations = 1)

    morphed_image = cv2.morphologyEx(dilated_image, cv2.MORPH_CLOSE, kernel)

    rgb_img = cv2.cvtColor(morphed_image, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    ret, binary_img = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY_INV)
    edged = cv2.Canny(binary_img, 30, 200)

    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    print("Number of Contours found = " + str(len(contours)))
    return contours


def rotate_cont(cnt, angle):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]

    coordinates = cnt_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)
    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    thetas = np.deg2rad(thetas)

    xs, ys = pol2cart(thetas, rhos)
    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys

    cnt_rotated = cnt_norm + [cx, cy]
    cnt_rotated = cnt_rotated.astype(np.int32)
    return cnt_rotated


def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


if __name__ == '__main__':
    main()


