import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import floor

# A matrix de homografia que montamos possui oito números que temos que descobrir, e um que é constante,
# totalizando 9 elementos em uma matriz 3x3.
# Para calcular esses 8 elementos, vamos montar um sistema com 8 equações; por isso precisamos de quatro pontos
# da imagem (cada ponto possui 2 coordenadas, totalizando 4*2 = 8 números que usaremos nas equações)
def make_matrix(src, dst):
    assert len(src) == 4, "forneça exatamente 4 pontos"
    assert len(src) == len(dst), "src e dst devem ter o mesmo tamanho"

    (s1x, s1y), (s2x, s2y), (s3x, s3y), (s4x, s4y) = src
    (d1x, d1y), (d2x, d2y), (d3x, d3y), (d4x, d4y) = dst

    # As oito equações em forma de equação linear (https://en.wikipedia.org/wiki/Matrix_(mathematics)#Linear_equations)
    A = np.array([
        [s1x, s1y, 1, 0  , 0  , 0, -d1x*s1x, -d1x*s1y, -d1x],
        [0  , 0  , 0, s1x, s1y, 1, -d1y*s1x, -d1y*s1y, -d1y],
        [s2x, s2y, 1, 0  , 0  , 0, -d2x*s2x, -d2x*s2y, -d2x],
        [0  , 0  , 0, s2x, s2y, 1, -d2y*s2x, -d2y*s2y, -d2y],
        [s3x, s3y, 1, 0  , 0  , 0, -d3x*s3x, -d3x*s3y, -d3x],
        [0  , 0  , 0, s3x, s3y, 1, -d3y*s3x, -d3y*s3y, -d3y],
        [s4x, s4y, 1, 0  , 0  , 0, -d4x*s4x, -d4x*s4y, -d4x],
        [0  , 0  , 0, s4x, s4y, 1, -d4y*s4x, -d4y*s4y, -d4y],
        [0  , 0  , 0, 0  , 0  , 0,  0      , 0       , 1]
    ])
    # O último número define o valor de h33. Mude-o à seu bel prazer (menos pra 0!)
    b = [0, 0, 0, 0, 0, 0, 0, 0, 12.34]

    (h11, h12, h13, h21, h22, h23, h31, h32, h33) = np.linalg.solve(A, b)
    return np.array([
        [h11, h12, h13],
        [h21, h22, h23],
        [h31, h32, h33],
    ])

# Aproxima um pixel via nearest-neighbour
def get_pixel(img, x: float, y: float):
    x = floor(x)
    y = floor(y)

    # Testa se o pixel pedido existe
    ny, nx, _ = img.shape
    if x >= nx or y >= ny or x < 0 or y < 0:
        return [0, 0, 0]

    return img[y, x]

# Aplica transformação vinda de uma imagem "source" para uma imagem "destination"
def apply_matrix(src, dst, matrix):
    H, W, _ = dst.shape

    for y in range(0, H):
        for x in range(0, W):
            px, py, pw = np.matmul(matrix, np.array([x, y, 1]))
            # A transformação pode resultar em posições com casas decimais.
            # Usamos interpolação para calcular os pixeis em posições não-discretas
            dst[y, x] = get_pixel(src, (px/pw), (py/pw))


def warp(source_image, source_points, destination_points, out_w, out_h):
    # Calcula a transformação
    h = make_matrix(source_points, destination_points)
    h = np.linalg.inv(h)

    # Cria uma nova imagem, onde escrevemos o resultado
    dst = np.zeros([out_h, out_w, 3], dtype=np.uint8)
    apply_matrix(source_image, dst, h)

    return dst