import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_homography_matrix(source, destination):
    assert len(source) == 4, "must provide more than 4 source points"
    assert len(source) == len(destination), "source and destination must be of equal length"

    N = len(source)
    A = []
    b = []
    for i in range(N):
        s_x, s_y = source[i]
        d_x, d_y = destination[i]
        A.append([s_x, s_y, 1, 0, 0, 0, (-d_x)*(s_x), (-d_x)*(s_y)])
        A.append([0, 0, 0, s_x, s_y, 1, (-d_y)*(s_x), (-d_y)*(s_y)])
        b += [d_x, d_y]
    A = np.array(A)
    h = np.linalg.lstsq(A, b, rcond=None)[0]
    h = np.concatenate((h, [1]), axis=-1)
    return np.reshape(h, (3, 3))

if __name__ == "__main__":
    out_w = 2048
    out_h = 1536

    source_points = np.array([
        [836,  1070],
        [2652,  665],
        [ 799, 2465],
        [2836, 2460]
    ])
    destination_points = np.array([
        [0, 0],
        [out_w, 0],
        [0, out_h],
        [out_w, out_h]
    ])
    source_image = cv2.imread("foto1_cap1.jpg")
    t_source_image = source_image.copy()

    # draw markings on the source image
    for i, pts in enumerate(source_points):
        # cv2.putText(source_image, str(i+1), (pts[0] + 15, pts[1]), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 5)
        cv2.circle(source_image, pts, 10, (0, 0, 255), 20)

    h = get_homography_matrix(source_points, destination_points)
    destination_image = cv2.warpPerspective(t_source_image, h, (out_w, out_h))

    figure = plt.figure(figsize=(12, 6))

    subplot1 = figure.add_subplot(1, 2, 1)
    subplot1.title.set_text("Source Image")
    subplot1.imshow(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))

    subplot2 = figure.add_subplot(1, 2, 2)
    subplot2.title.set_text("Destination Image")
    subplot2.imshow(cv2.cvtColor(destination_image, cv2.COLOR_BGR2RGB))

    # plt.show()
    plt.savefig("output.png")