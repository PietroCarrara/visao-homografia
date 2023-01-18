import cv2
import sys
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
    for point in src_points:
        cv2.circle(src_copy, point, 10, (0, 0, 255), 20)

    size = (floor(h*scale), floor(w*scale))
    cv2.imshow("input", cv2.resize(src_copy, size))

    if len(src_points) < 4:
        cantos = [
            "superior esquerdo",
            "superior direito",
            "inferior esquerdo",
            "inferior direito",
        ]
        print(f"Clique no canto {cantos[len(src_points)]}")

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

    min_x = min(src_points[0][0], src_points[2][0])
    max_x = max(src_points[1][0], src_points[3][0])
    min_y = min(src_points[0][1], src_points[1][1])
    max_y = max(src_points[2][1], src_points[3][1])

    OUT_W = max_x - min_x
    OUT_H = max_y - min_y

    dst = np.array([
        [0, 0],
        [OUT_W, 0],
        [0, OUT_H],
        [OUT_W, OUT_H]
    ])

    res = warp(src, src_points, dst, OUT_W, OUT_H)
    cv2.imwrite("output.png", res)

    w, h, _ = res.shape
    scale = min(600/h, 600/w)
    size = (floor(h*scale), floor(w*scale))
    cv2.imshow("output", cv2.resize(res, size))

FILENAME = "foto1_cap1.jpg" if len(sys.argv) < 2 else sys.argv[1]

src = cv2.imread(FILENAME)
src_points = []
scale = 1

update_input()
cv2.setMouseCallback("input", onclick)

while True:
    k = cv2.waitKey() & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
        break
