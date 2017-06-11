import numpy as np

results = np.zeros((10, 5))

print(results)

results[0, [1, 2, 3]] = 1

print(results)
