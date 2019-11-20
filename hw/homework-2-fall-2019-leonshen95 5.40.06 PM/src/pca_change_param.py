import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import time;


# import data.
X = np.load("mnist_data.npy")
y = np.load("mnist_labels.npy")

# shuffle and split into training and test set
sample_range = range(3000, 21000, 3000)
compo_range = range(50, 750, 100)
time_consume_sample =[]
time_consume_comp = []

# change in sampling range
for size in sample_range:
    start_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=int(size*0.8), test_size=int(size*0.2), random_state=0, shuffle=True)

    scaler = StandardScaler(with_std=False)
    # Fit on training set only.
    scaler.fit(X_train)

    # Apply transform to both the training set and the test set.
    train_img = scaler.transform(X_train)
    test_img = scaler.transform(X_test)

    # Make an instance of the Model
    pca = PCA(n_components=300)

    # Fit data on the model
    pca.fit(train_img)

    X_train = pca.transform(train_img)
    X_test = pca.transform(test_img)

    # import knn classifier
    classifier = KNeighborsClassifier(n_neighbors=5)

    # Train the model on the data
    classifier.fit(X_train, y_train)

    time_consume_sample.append(time.time() - start_time)

# change in component range
for n in compo_range:
    start_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, shuffle=True)

    scaler = StandardScaler(with_std=False)
    # Fit on training set only.
    scaler.fit(X_train)

    # Apply transform to both the training set and the test set.
    train_img = scaler.transform(X_train)
    test_img = scaler.transform(X_test)

    # Make an instance of the Model
    pca = PCA(n_components=n)

    # Fit data on the model
    pca.fit(train_img)

    X_train = pca.transform(train_img)
    X_test = pca.transform(test_img)

    # import knn classifier
    classifier = KNeighborsClassifier(n_neighbors=5)

    # Train the model on the data
    classifier.fit(X_train, y_train)

    time_consume_comp.append(time.time() - start_time)


# plot graph with different sampling number
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(time_consume_sample, sample_range, color = color)
ax1.set_xlabel('time (s)')
ax1.set_ylabel('sample_range', color=color)
ax1.tick_params(axis='y', labelcolor=color)

# plot graph with different component number
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('component_range', color=color)
ax2.plot(time_consume_comp, compo_range, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()

