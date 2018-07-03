import numpy as np
import random
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.misc import imsave

# learning rate
ALPHA = 0.5
# discount factor
GAMMA = 0.99
# number of epochs for training
  = 101
# maximum number of steps per epoch
MAXIMUM_STEPS = 500

'''
A - position of the agent
* - destination
X - wall
0 - empty cell
'''
INITIAL_MAZE = []
maze_matrix = np.loadtxt('maze_matrix.txt', dtype='int', delimiter=', ')
np.mat(maze_matrix)

for i in range(len(maze_matrix)):
    temp = []
    for j in range(len(maze_matrix[0])):
        if maze_matrix[i, j] == 1:
            temp.append('X')
        if maze_matrix[i, j] == 0:
            temp.append('0')
    INITIAL_MAZE.append(temp)
INITIAL_MAZE[57][103] = '*'
# INITIAL_MAZE[1][1] = 'A'
# np.savetxt('INITIAL_MAZE.txt', INITIAL_MAZE, fmt='%s', delimiter=', ')


'''
Possible moves
L - left
U - up
R - right
D - down
'''
MOVES = ['L', 'U', 'R', 'D']


class Maze:
    def __init__(self, q_table=None):
        self.maze = deepcopy(INITIAL_MAZE)
        self.maze_matrix = maze_matrix*255
        self.final_state = self.get_position('*')  # position of the destination
        self.Q = q_table

    def get_position(self, symbol):
        for ii in range(0, len(self.maze)):
            for jj in range(0, len(self.maze[ii])):
                if self.maze[ii][jj] == symbol:
                    return ii, jj

    def select_move(self, epsilon):
        """
        Selects the next move.
        It may be the one having the greatest Q value or a random one (allowing exploration of new paths)
        :param epsilon: exploration factor [0, 1]
        for values close to 1 it's more likely to choose a random action
        for values close to 0 it's more likely to choose the best move given by the Q table
        :return: next move (one of L, U, R, D)
        """
        if random.random() > 1 - epsilon:
            return random.choice(self.get_possible_moves())
        else:
            return self.get_best_q()

    def get_best_q(self):
        maximum = -float("inf")
        p = self.get_position('A')  # position of the agent
        best_moves = []
        for m in self.get_possible_moves():
            if self.Q[p, m] == maximum:
                best_moves.append(m)
            if self.Q[p, m] > maximum:
                maximum = self.Q[p, m]
                best_moves = [m]
        return random.choice(best_moves)  # one of the best Q's for current state

    def get_reward(self):
        """
        Compute the reward for the current state
        :return: 1 if the agent has reached the final state, -1 otherwise
        """
        return -1 if self.get_position('A') != self.final_state else 1000

    def get_possible_moves(self):
        """
        :return: list containing all the possible moves from the current state
        """
        p = self.get_position('A')
        moves = deepcopy(MOVES)

        # remove invalid moves
        if p[1] == 0 or self.maze[p[0]][p[1] - 1] == 'X':
            moves.remove('L')

        if p[0] == 0 or self.maze[p[0] - 1][p[1]] == 'X':
            moves.remove('U')

        if p[1] == len(self.maze[p[0]]) - 1 or self.maze[p[0]][p[1] + 1] == 'X':
            moves.remove('R')

        if p[0] == len(self.maze) - 1 or self.maze[p[0] + 1][p[1]] == 'X':
            moves.remove('D')

        return moves

    def update_maze(self, move):
        p = self.get_position('A')
        self.maze[p[0]][p[1]] = '0'  # old position of the agent

        # new position based on the move
        if move == 'U':
            self.maze[p[0] - 1][p[1]] = 'A'

        if move == 'D':
            self.maze[p[0] + 1][p[1]] = 'A'

        if move == 'L':
            self.maze[p[0]][p[1] - 1] = 'A'

        if move == 'R':
            self.maze[p[0]][p[1] + 1] = 'A'

    def training(self):
        """
        Performs training in order to find optimal values for the Q table
        """"""
        第一次运行程序运行下面注释
        """"""
        self.Q = {}
        for i in range(0, len(INITIAL_MAZE)):
            for j in range(0, len(INITIAL_MAZE[i])):
                for k in MOVES:
                    self.Q[(i, j), k] = 0
        """

        epsilon = 0.8  # allow more exploration in the beginning of the training
        flag = 0
        print("TRAIN:")

        for epochs in range(EPOCHS):
            if epochs % 10 == 0:
                print("epochs %d, successful times %d" % (epochs, flag))
            self.maze = deepcopy(INITIAL_MAZE)
            # a = random.randrange(55, max(0, int(54-0.058*epochs)), -2)
            # b = random.randrange(101, max(0, int(100-0.103*epochs)), -2)
            a = random.randrange(1, 58, 2)
            b = random.randrange(1, 104, 2)
            print("(%d, %d)" % (a, b))
            self.maze[a][b] = 'A'
            s = self.get_position('A')
            steps = 0
            while (s != self.final_state) and steps < MAXIMUM_STEPS:
                steps += 1
                next_move = self.select_move(epsilon)
                self.update_maze(next_move)

                r = self.get_reward()
                new_p = self.get_position('A')  # new position of the agent
                best_q = self.get_best_q()

                # update Q table using the TD learning rule
                self.Q[s, next_move] += ALPHA * (r + GAMMA * self.Q[new_p, best_q] - self.Q[s, next_move])

                s = self.get_position('A')
                epsilon -= (epsilon * 0.001)  # decay the exploration factor
                if s == self.final_state:
                    flag += 1
                    self.maze_matrix[a][b] = 0
                else:
                    self.maze_matrix[a][b] = 160
            # if epochs % 20 == 19:
        f_temp = open('Q_copy.txt', 'w')
        f_temp.write(str(self.Q))
        f_temp.close()
        """
        plt.imshow(self.maze_matrix, cmap='gray_r')
        plt.axis('off')
        plt.show()
        """
        print("TRAINING END!")

    def test(self):
        print("TEST")
        self.maze = deepcopy(INITIAL_MAZE)
        # self.maze[0][1] = 'A'
        # self.print_maze()
        """
        aa = random.randrange(57, 1, -2)
        bb = random.randrange(103, 1, -2)
        print("(%d, %d)" % (aa, bb))
        """
        for aa in range(1, 58, 2):
            for bb in range(1, 104, 2):
                self.maze[aa][bb] = 'A'
                s = self.get_position('A')
                steps = 0
                while (s != self.final_state) and steps < MAXIMUM_STEPS:
                    steps += 1
                    self.update_maze(self.select_move(epsilon=0))
                    s = self.get_position('A')
                    # print(s)  # self.print_maze()
                print("Agent reached (%d, %d) in %d steps" % (aa, bb, steps))
                # self.maze[s[0]][s[1]] = '0'
                self.maze = deepcopy(INITIAL_MAZE)
                if steps == MAXIMUM_STEPS:
                    self.maze_matrix[aa][bb] = 160
                else:
                    self.maze_matrix[aa][bb] = 0

        plt.imshow(self.maze_matrix, cmap='gray_r')
        plt.axis('off')
        plt.show()
        imsave('self_maze_matrix_copy.jpg', 1 - self.maze_matrix)

    def print_maze(self):
        for element in self.maze:
            print(element)
        print()


""""""
f = open('Q_copy.txt', 'r')  # 第一次运行将此行注释掉
Q = f.read()    # 第一次运行将此行注释掉
f.close()   # 第一次运行将此行注释掉
maze = Maze(q_table=eval(Q))    # 第一次运行将此行注释掉

# maze = Maze()    # 第一次运行时运行此行
# plt.imshow(maze_matrix, cmap='gray_r')
# plt.axis('off')
# plt.show()
# maze.training()
"""
for _ in range(2000):
    print("现在在 %d" % _)
    maze.training()
"""
maze.test()
""""""
