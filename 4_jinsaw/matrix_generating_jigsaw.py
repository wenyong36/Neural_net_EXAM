import random
import numpy as np


def matrix_jigsaw():
    matrix = np.zeros([64, 64], int)
    x = []
    n = list(range(64))
    for i in range(64):
        x.append(random.choice(n))
        n.remove(x[i])
    for i in range(64):
        matrix[x[i]][i] = 1

    return matrix


def image_cut(pic):
    pic_temp = np.zeros((64, 28, 28, 3), dtype=int)
    index = list(range(64))
    for i in range(8):  # 第i行
        a = pic[[range(28 * i, 28 * i + 28, 1)]]
        for j in range(8):
            k = random.choice(index)
            for l in range(28):
                pic_temp[[k], l] = a[l, [range(28*j, 28*j+28, 1)]]
            index.remove(k)
    return pic_temp

# A = matrix_jigsaw()
# np.savetxt('A_jigsaw.txt', A, fmt='%d', delimiter=', ')

