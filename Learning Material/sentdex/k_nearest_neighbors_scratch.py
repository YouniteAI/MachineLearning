import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter

style.use('fivethirtyeight')

dataset = {
    'k': [[1, 2], [2, 3], [3, 1]],
    'r': [[6, 5], [7, 7], [8, 6]],
}
new_features = [4, 1]


def k_nearest_neighbors(data, new_data, k=3):
    if len(data) >= k:
        warnings.warn('k is set to a value less than total voting groups!')

    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(new_data))
            distances.append([euclidean_distance, group])

    votes = []
    for i in sorted(distances)[:k]:
        votes.append(i[1])
    print(votes)

    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result


result = k_nearest_neighbors(dataset, new_features, k=3)
print(result)

for i in dataset:
    for j in dataset[i]:
        plt.scatter(j[0], j[1], s=100, color=i)

plt.scatter(new_features[0], new_features[1], color=result)
plt.show()





