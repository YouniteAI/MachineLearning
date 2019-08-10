'''

Creating K_Means from Scratch

K means Algorithm

- pick any K data points as starting centroids
- measure distances of every other point to the centroids
- the points closest to a centroid are classified by that centroid
- calculate the mean of the classified points and find a new centroid
- repeat

'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing
import pandas as pd
style.use('ggplot')
pd.options.display.max_columns = 20

X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11], [1, 3],
                     [5, 4],
                     [6, 4],
                     [9, 9],
                     [10, 10]])

plt.scatter(X[:, 0], X[:, 1], s=100)

colors = 10*['g', 'r', 'c', 'b', 'k']


class K_Means:

    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for j in range(self.k):
                self.classifications[j] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                pass
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid)/original_centroid * 100.0) > self.tol:
                    optimized = False

            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


# https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls
df = pd.read_excel('titanic.xls')
df.drop(['body', 'name'], 1, inplace=True)
# df.convert_objects(convert_numeric=True)
print(df.head())
df.fillna(0, inplace=True)


def handle_non_numerical_data(df):
    # handling non-numerical data: must convert.
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        # print(column,df[column].dtype)
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:

            column_contents = df[column].values.tolist()
            # finding just the uniques
            unique_elements = set(column_contents)
            # great, found them.
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    # creating dict that contains new
                    # id per unique string
                    text_digit_vals[unique] = x
                    x += 1
            # now we map the new "id" vlaue
            # to replace the string.
            df[column] = list(map(convert_to_int, df[column]))

    return df


df = handle_non_numerical_data(df)

# add/remove features just to see impact they have.
df.drop(['ticket', 'home.dest'], 1, inplace=True)
print(df.head())

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = K_Means()
clf.fit(X)

correct = 0
for i in range(len(X)):

    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction == y[i]:
        correct += 1

print(correct / len(X))