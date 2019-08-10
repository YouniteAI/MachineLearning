import numpy as np
import pandas as pd
from math import sqrt
import warnings
from collections import Counter
import random

pd.options.display.max_rows = 5
pd.options.display.max_columns = 15


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

    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1]/k

    return vote_result, confidence


accuracies = []

for i in range(25):
    df = pd.read_csv('breast-cancer-wisconsin.data')
    df.replace('?', -99999, inplace=True)
    df.drop('id', 1, inplace=True)
    full_data = df.astype(float).values.tolist()        # convert all the data to floats
    random.shuffle(full_data)

    test_size = 0.4
    train_set = {
        2: [],
        4: []
    }
    test_set = {
        2: [],
        4: []
    }

    train_data = full_data[:-int(test_size*len(full_data))]     # up until last 20%
    test_data = full_data[-int(test_size*len(full_data)):]  # last 20%

    for i in train_data:
        train_set[i[-1]].append(i[:-1])
    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0

    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbors(train_set, data, k=5)
            if group == vote:
                correct += 1
            total += 1
    accuracies.append(correct/total)

print(sum(accuracies)/len(accuracies))