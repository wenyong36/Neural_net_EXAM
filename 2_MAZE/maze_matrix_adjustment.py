import matplotlib.pyplot as plt
import numpy as np

f = open('Q_copy.txt', 'r')  # 第一次运行将此行注释掉
Q_ = f.read()    # 第一次运行将此行注释掉
f.close()   # 第一次运行将此行注释掉
Q = eval(Q_)

maze_matrix = np.loadtxt('maze_matrix.txt', dtype='int', delimiter=', ')
np.mat(maze_matrix)

MOVES = ['L', 'U', 'R', 'D']

for i in range(59):
    for k in MOVES:
        Q[(i, 0), k] = -1000
        Q[(i, 104), k] = -1000

for j in range(105):
    for k in MOVES:
        Q[(0, j), k] = -1000
        Q[(58, j), k] = -1000

for i in range(57):
    for j in range(103):
        if maze_matrix[i][j+1] == 1:
            Q[(i+1, j+1), 'U'] = -1000
        if maze_matrix[i+2][j+1] == 1:
            Q[(i+1, j+1), 'D'] = -1000
        if maze_matrix[i+1][j] == 1:
            Q[(i+1, j+1), 'L'] = -1000
        if maze_matrix[i+1][j+2] == 1:
            Q[(i+1, j+1), 'R'] = -1000

for i in range(30):
    for j in range(53):
        for k in MOVES:
            Q[(2*i, 2*j), k] = -1000

f_temp = open('Q_copy.txt', 'w')
f_temp.write(str(Q))
f_temp.close()
