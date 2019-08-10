'''

NOTE: This is a very simplified view on how to handle data, for categorical data we should be using
a one-hot-encoding for more interesting results from our models.

'''

import matplotlib as plt
from matplotlib import style
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
style.use('ggplot')
pd.options.display.max_columns = 20

df = pd.read_excel('titanic.xls')
df.drop(['body', 'name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)


def handle_non_numerical_data(df):
    columns = df.columns.values     # list of all column names

    for column in columns:
        text_digit_vals = {}        # set which contains mapping from categorical -> numerical

        def convert_to_int(text):
            '''
            :param text:
            :return: numerical
            '''
            numerical = text_digit_vals[text]
            return numerical

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:  # if the column is not a number
            column_contents = df[column].values.tolist()    # put the samples from that column in a list
            unique_elements = set(column_contents)  # takes the set, or the unique values of the list
            x = 0
            for unique in unique_elements:  # for every unique value, allow it to correspond to a unique number
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1
            df[column] = list(map(convert_to_int, df[column]))  # transform the columns to integers using helper
                                                                # function
    return df


df = handle_non_numerical_data(df)
df.drop(['boat', 'sex'], 1, inplace=True)
print(df.head())

X = np.array(df.drop(['survived'], 1)).astype(float)
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print()
print(correct/len(X))














































