import numpy as np

def impute_missing(X):
    for i in range(len(X)):
        if X[i] == 'nan':
            X[i] = np.median(X)
    return X

# Because mean is the result of a model over the "errors".
# One is more likely to expect a common center.
# Thus a median is preferred as the center because it disregards the two extremes at the sides.
