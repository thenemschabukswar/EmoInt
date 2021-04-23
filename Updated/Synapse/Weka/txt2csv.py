# importing panda and csv library
import pandas as pd
import csv

# readinag given csv file and creating dataframe
data = pd.read_csv("joy_test.txt", delimiter = '\t')

# storing this dataframe in a csv file
data.to_csv('joy_test.csv',index = None,quoting=csv.QUOTE_NONNUMERIC, escapechar="\\" )

