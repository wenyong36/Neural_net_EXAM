import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.misc import imsave
from maze_learning import Maze


f = open('Q_copy.txt', 'r')
Q = f.read()
f.close()
maze = Maze(q_table=eval(Q))
path, steps = maze.maze_path()


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

for p in path:
    if p[0] % 2 == 1:
        if p[1] % 2 == 1:
            for i in range(17):
                for j in range(17):
                    maze_show[(p[0]-1)*10+5+i][(p[1]-1)*10+5+j] = 50
        else:
            for i in range(17):
                for j in range(3):
                    maze_show[(p[0] - 1)*10 + 5 + i][p[1] * 10 + 2 + j] = 50
    else:
        for i in range(3):
            for j in range(17):
                maze_show[p[0] * 10 + 2 + i][(p[1] - 1) * 10 + 5 + j] = 50
    
print("到达指定地点需要 %d 步" %int(steps/2))
plt.imshow(maze_show, cmap='gray_r')
plt.axis('off')
plt.show()

