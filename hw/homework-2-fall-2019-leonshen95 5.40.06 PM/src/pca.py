import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# import data.
X = np.load("mnist_data.npy")
y = np.load("mnist_labels.npy")

# shuffle and split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, shuffle=True)

scaler = StandardScaler(with_std=False)
# Fit on training set only.
scaler.fit(X_train)


# Apply transform to both the training set and the test set.
train_img = scaler.transform(X_train)
test_img = scaler.transform(X_test)

# Make an instance of the Model
pca = PCA(n_components=100)

# Fit data on the model
pca.fit(train_img)

# plot first 10 principle component images
fig, ax = plt.subplots(1, 10, figsize=(9,4),
                       subplot_kw={'xticks':[], 'yticks':[]},
                       gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i in range(10):
    ax[i].imshow(train_img[i].reshape(28, 28), cmap='binary_r')

ax[0].set_ylabel('reconstruction of image')

plt.show()

X_train = pca.transform(train_img)
X_test = pca.transform(test_img)

# plot cdf graph
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.title("CDF of explained variance")
# plt.xlabel('number of components')
# plt.ylabel('explained variance')
# plt.show()

# import knn classifier
classifier = KNeighborsClassifier(n_neighbors=5)

# Train the model on the data
classifier.fit(X_train, y_train)

# Predict the labels on the entire test set.
# prediction = classifier.predict(X_test)
#
# # Obtain an accuracy for training
# score_train = classifier.score(X_train, y_train)
# print("training accuracy: " + str(score_train))
#
# # Obtain an accuracy for entire testing
# score_test = classifier.score(X_test, y_test)
# print("test accuracy: " + str(score_test))



