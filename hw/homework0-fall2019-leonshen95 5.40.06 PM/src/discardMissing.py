def discard_missing(X):
    X = filter(lambda x: x is not 'nan', X)
    return X