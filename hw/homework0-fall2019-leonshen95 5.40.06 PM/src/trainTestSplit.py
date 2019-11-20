import random


# make sure input t_f as fraction that is less than 1.


def train_test_split(X, y, t_f):
    x_num = int(t_f * len(X))  # for example: num is the total random nums of data in X training set
    y_num = int(t_f * len(y))
    shuffle_x = random.shuffle(X)
    shuffle_y = random.shuffle(y)
    X_train = shuffle_x[:x_num]  # first t_f fraction of shuffled list
    y_train = shuffle_y[:y_num]
    X_test = shuffle_x[x_num:]  # last 1-t_f traction of shuffled list
    y_test = shuffle_y[y_num:]
    return X_train, y_train, X_test, y_test