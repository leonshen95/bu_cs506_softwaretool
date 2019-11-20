import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# import data.
X = np.load("mnist_data.npy")
y = np.load("mnist_labels.npy")

# shuffle and split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, shuffle=True)

train_scores = []
test_scores =[]
k_range = range(1, 25, 2)
for k in k_range:
    # Use knn classifier
    classifier = KNeighborsClassifier(n_neighbors=k)

    # Train the model on the data
    classifier.fit(X_train, y_train)

    # Predict the labels on the entire test set.
    prediction = classifier.predict(X_test)

    # Obtain an accuracy for training
    score_train = classifier.score(X_train, y_train)
    train_scores.append(score_train)

    # Obtain an accuracy for entire testing
    score_test = classifier.score(X_test, y_test)
    test_scores.append(score_test)

plt.plot(k_range, train_scores, 'r', label='train_acc')
plt.plot(k_range, test_scores, 'g', label='test_acc')
plt.xlabel('Value of K for KNN')
plt.ylabel('Accuracy')
plt.legend(loc='upper right')
plt.show()