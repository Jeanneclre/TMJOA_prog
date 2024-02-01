import pandas as pd
import numpy as np
import csv


A = pd.read_csv("./TMJOAI_Long_040422_Norm.csv")

y = A.iloc[:, 0].values
X = A.iloc[:, 1:].values
# Read the csv file and get the header of the file
with open('./TMJOAI_Long_040422_Norm.csv', newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    header = next(csvreader)

nb_0=0
nb_1=0
for i in y:
    if i == 0:
        nb_0 += 1
    else:
        nb_1 += 1

print('class 0:', nb_0)
print('class 1:',nb_1)

#For each column, calculate the mean of the row of the X matrix
# list with the mean of each column
mean = np.mean(X,axis=0)
print('shape mean:', mean.shape)
print('mean:', mean)

# create a file with the name of each column and their mean
# the file must have the name of the feature in each column. There are only 2 rows
# the first row is the name of the feature
# the second row is the mean of the feature
with open('mean.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(header)
    csvwriter.writerow(mean)