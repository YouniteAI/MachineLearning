import numpy as np

x = np.ones(784)
print(x)

x = np.reshape(x, newshape=[28, 28])

print(x)
print(len(x))
