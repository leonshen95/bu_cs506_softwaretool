import numpy as np


# get data from user and append the data into X and y

def import_data(filename):
    X = []
    y = []
    file = open(filename)
    for data in file:
        value = data.strip().split(',')
        processed_data = list(map(change_question_mark, value))
        att, cor_class = processed_data[:278], int(value[279])
        X.append(att)
        y.append(cor_class)
    return X,y


# change the question mark into nah

def change_question_mark(string):
    if string == '?':
        string = np.NaN
    return float(string)
