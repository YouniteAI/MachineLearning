'''

pandas is a column orientated data analysis API. Good for handling and analyzing input data

'''

# imports the pandas API and prints the API version

import pandas as pd
import matplotlib.pyplot as mpl
import numpy as np

# print(pd.__version__)


# Primary data structure in pandas are implemented as two classes:
# Series - a single column
# DataFrame - contains one or more Series and a name for each
# > data frames are commonly used abstractions for data manipulation

# Create Series
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
# print(city_names)
population = pd.Series([4553654, 565464, 546456])
# print(population)

# Create a DataFrame => dictionary mapping of name of Series to Series object

df = pd.DataFrame({
    'City Name': city_names,
    'Population': population
})
'''print(df)'''
# Most of the time we want to load an entire file into a data frame, so we can use the data

california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train."
                                           "csv", sep=",")
pd.set_option("display.max_columns", 20)

# Print the raw data
'''print(california_housing_dataframe)'''

# Show interesting stats about the data frame (count, mean, std, min, 25%, 50%, 75%, max)
'''print(california_housing_dataframe.describe())'''

# prints the first few records of the data_frame
'''print(california_housing_dataframe.head())'''

# Display histogram of 'Housing Median Age'
# Recall: a histogram of a variable indicates the domain of the variable on the x axis and the frequency on the y axis
hist = california_housing_dataframe.hist('housing_median_age')
'''mpl.show()'''

# Accessing Data

cities = pd.DataFrame({
    'City Name': city_names,
    'Population': population,
})
# Given a key in the data frame dictionary, the value is a Series
'''
print(type(cities['City Name']))
print(cities['City Name'])
'''

# cities['City Name'] refers to the Series, cities['City Name'][i] refers to the i-th element in the series
'''
print(cities['City Name'][1])
'''

# cities[n:m] accesses the sub-dataframe only containing the rows from n to m
'''
print(cities[1:3])
'''

# You can do operations on series
'''
print(population)
print(population/1000)
print(np.log(population))
'''

# You can use Series.apply(lamda_fnc) for more complex transformation on a Series, accepts a lamba function which
# applies to each value
'''
print(population.apply(lambda val: val > 1000000))
'''

# Modifying Data Frames, adding new Series to the DataFrame

cities['Area square miles'] = pd.Series([46, 87, 176.54, 12])
cities['Population Density'] = cities['Population']/cities['Area square miles']

'''
print(cities)
'''

# Exercise #1

'''

## Exercise #1

Modify the `cities` table by adding a new boolean column that is True if and only if *both* of the following are True:

  * The city is named after a saint.
  * The city has an area greater than 50 square miles.

**Note:** Boolean `Series` are combined using the bitwise, rather than the traditional boolean, operators. For example,
 when performing *logical and*, use `&` instead of `and`.

**Hint:** "San" in Spanish means "saint."

'''

# cities['named after saint and large'] = ((cities['Area square miles'] > 50) &
                                         #cities['City Name'].apply(lambda name: name.startswith('San')))
# print(cities)


'''

Indexes:
There are indexes on the side of the Series or DataFrame, start => 0, stop => last_index + 1, step => 1


'''

print(cities)

print(cities.index)

# re-index a data frame
print(cities.reindex([2, 0, 1]))

# randomize the re-indexing of a data frame
print(cities.reindex(np.random.permutation(cities.index)))

# Re-indexing a index that is not contained in the data frame:
print(cities.reindex([2, 5, 0, 1]))
print(cities.reindex([2, 5, 6, 0]))


















