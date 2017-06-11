import numpy as np
x = np.array(12)
print(x.ndim)

x = np.array([12, 3, 6, 14])
print(x.ndim)

x = np.array([[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]])
print(x.ndim)
print(x.shape)

x = x.reshape(5, 3)
print(x.ndim)
print(x.shape)
print(x)

x = x.reshape(5 * 3)
print(x)
x = x.astype('float32') * 2
print(x)