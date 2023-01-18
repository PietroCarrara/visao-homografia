import cv2
import numpy as np
from main import warp
from math import floor

def update_input():
    global src
    global src_points
    global scale

    w, h, _ = src.shape
    scale = min(600/h, 600/w)

    src_copy = src.copy()
    for i, point in enumerate(src_points):
        cv2.circle(src_copy, point, 10, (0, 0, 255), 20)

    size = (floor(h*scale), floor(w*scale))
    cv2.imshow("input", cv2.resize(src_copy, size))

def onclick(event, x, y, *_):
    global src_points
    global scale

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(src_points) >= 4:
            src_points = []

        src_points.append((floor(x / scale), floor(y / scale)))

        update_input()

        if len(src_points) == 4:
            show_result()

def show_result():
    global src
    global src_points

    dst = np.array([
        [0, 0],
        [OUT_W, 0],
        [0, OUT_H],
        [OUT_W, OUT_H]
    ])


    res = warp(src, src_points, dst, OUT_W, OUT_H)
    w, h, _ = src.shape
    scale = min(600/h, 600/w)
    size = (floor(h*scale), floor(w*scale))
    cv2.imshow("output", cv2.resize(res, size))

OUT_W = 2048
OUT_H = 1536

src_points = []
src = cv2.imread("foto1_cap1.jpg")
scale = 1

update_input()
cv2.setMouseCallback("input", onclick)

while True:
    k = cv2.waitKey() & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
        break
