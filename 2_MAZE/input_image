import matplotlib.pyplot as plt     # plt 用于显示图片
import matplotlib.image as mpimg    # mpimg 用于读取图片
import numpy as np
from scipy.misc import imsave

maze = mpimg.imread('maze.jpg')     # 此时 maze 就已经是一个 np.array 了，可以对它进行任意处理
maze = np.delete(maze, range(504, 449, -1), axis=0)
# print(maze.shape)
'''
# 转换成灰度图
maze_gray = 0.333*maze[:, :, 0]+0.334*maze[:, :, 1]+0.333*maze[:, :, 2]
# np.savetxt('maze.txt', maze_gray, fmt='%d', delimiter=', ')

maze_hat = np.zeros((450, 800))
# 0，1化
for i in range(450):
    for j in range(800):
        if maze_gray[i, j] < 201:
            maze_hat[i, j] = 255
        elif maze_gray[i, j] > 200:
            maze_hat[i, j] = 0

# np.savetxt('maze_hat.txt', maze_hat, fmt='%d', delimiter=', ')


# 显示图片
plt.subplot(221)
plt.imshow(maze)
plt.axis('off')
plt.subplot(222)
plt.imshow(maze_gray, cmap='gray_r')
plt.axis('off')
plt.subplot(223)
plt.imshow(maze_gray, cmap='gray')
plt.axis('off')
plt.subplot(224)
''''''
plt.imshow(maze_hat, cmap='gray_r')
plt.axis('off')
plt.show()
'''
'''
# 转换成0，1矩阵
maze_matrix = np.ones((59, 105))
for i in range(28):
    for j in range(52):
        maze_matrix[2*i+1, 2*j+1] = 0
        m = 0
        for k in range(15):
            m = m + maze_hat[int(15.5*i+9+k), int(15.3*j+8)]+maze_hat[int(15.5*i+9+k), int(15.3*j+9)] + \
                maze_hat[int(15.5*i+9+k), int(15.3*j+10)]
        if m < 256:
            maze_matrix[2*i+2, 2*j+1] = 0

for i in range(29):
    for j in range(51):
        maze_matrix[2 * i + 1, 2 * j + 1] = 0
        n = 0
        for k in range(15):
            n = n + maze_hat[int(15.5 * i + 9), int(15.3*j+8+k)] + maze_hat[int(15.5 * i + 10), int(15.3*j+8+k)] + \
                maze_hat[int(15.5 * i + 11), int(15.3*j+8+k)]
        if n < 256:
            maze_matrix[2 * i + 1, 2 * j + 2] = 0
maze_matrix[57, 103] = 0

np.savetxt('maze_matrix.txt', maze_matrix, fmt='%d', delimiter=', ')
'''
maze_matrixrd = np.loadtxt('maze_matrix.txt', dtype='int', delimiter=', ')
np.mat(maze_matrixrd)
# print(maze_matrixrd)

maze_show = np.ones((587, 1047))*255
for i in range(29):
    for j in range(52):
        for m in range(17):
            for n in range(17):
                maze_show[5+20*i+m, 5+20*j+n] = 0
        if maze_matrixrd[2*i+1, 2*j+2] == 0:
            for k in range(17):
                maze_show[5 + 20 * i + k, 21 + 20 * j + 1] = 0
                maze_show[5 + 20 * i + k, 21 + 20 * j + 2] = 0
                maze_show[5 + 20 * i + k, 21 + 20 * j + 3] = 0
        if maze_matrixrd[2*i+2, 2*j+1] == 0:
            for k in range(17):
                maze_show[21 + 20 * i + 1, 5 + 20 * j + k] = 0
                maze_show[21 + 20 * i + 2, 5 + 20 * j + k] = 0
                maze_show[21 + 20 * i + 3, 5 + 20 * j + k] = 0

# plt.imshow(maze_show, cmap='gray_r')
# plt.axis('off')
# plt.show()

# imsave('maze_matrix.jpg', 1-maze_matrixrd)
plt.imshow(maze_matrixrd, cmap='gray_r')
plt.axis('off')
plt.show()
