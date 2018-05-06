import numpy as np

a = np.mat([[1, 2, 3], [3, 3, 4], [2, 4, 5]])
print(1 /2 * (a + a.T) + 1 /2 * (a - a.T))