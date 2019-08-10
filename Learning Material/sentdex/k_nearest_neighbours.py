import numpy as np
from sklearn import preprocessing, neighbors, svm
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
pd.options.display.max_rows = 5
pd.options.display.max_columns = 15

df = pd.read_csv('breast-cancer-wisconsin.data')

df.replace('?', -99999, inplace=True)
df.drop('id', 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = svm.SVC()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print(accuracy)


'''
print(accuracy)

example_measure = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 5, 3, 2, 3, 2, 1]])
example_measure = example_measure.reshape(len(example_measure), -1)

prediction = model.predict(example_measure)
print(prediction)
'''


