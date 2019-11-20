import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# import data.
X = np.load("mnist_data.npy")
y = np.load("mnist_labels.npy")

# shuffle and split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, shuffle=True)

# Use Logistic Regression with lbfgs and mutinomial
logisticRegr = LogisticRegression(solver="lbfgs", multi_class="multinomial")

# Train the model on the data
logisticRegr.fit(X_train, y_train)

# Predict the labels on the entire test set.
predictions = logisticRegr.predict(X_test)

# Obtain an accuracy for training
score_train = logisticRegr.score(X_train, y_train)
print("training accuracy: " + str(score_train))

# Obtain an accuracy for overall testing
score = logisticRegr.score(X_test, y_test)
print("test accuracy: " + str(score))