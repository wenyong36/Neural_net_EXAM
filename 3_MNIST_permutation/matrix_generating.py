import random
import numpy as np

x = []
n = list(range(784))
for i in range(784):
    x.append(random.choice(n))
    n.remove(x[i])
A = np.zeros([784, 784], int)
for i in range(784):
    A[x[i]][i] = 1
np.savetxt('A.txt', A, fmt='%d', delimiter=', ')
