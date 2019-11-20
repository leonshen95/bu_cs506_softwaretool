from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import folium
from sklearn import mixture
from folium.plugins import MarkerCluster

# default coordinate of the folium map
defeault_location=[40.693943, -73.985880]

# fields that need to be extracted from the original dataset
fields = ['latitude', 'longitude', 'price']

# extract and convert the data type accordingly
df = pd.read_csv('listings.csv', usecols=fields, dtype={"latitude": float, "longitude": float, "price": int})
# ignore price that is 0
df = df.loc[df.ne(0).all(axis=1)]
df_nyc = df
# scale the data
scaler = MinMaxScaler()
scaler.fit(df)
df = scaler.transform(df)

# print dataset content
print(df)

# print the size of dataset
print(df.size)

# K-means++ cluster
kmeans = KMeans(n_clusters=10).fit(df)
y_kmeans = kmeans.predict(df)
centroids = kmeans.cluster_centers_

# print centroids
print(centroids)

# plot clustering in a grid graph
plt.scatter(df[:, 0], df[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.grid()
plt.show()

# plot clustering using folium
locations = df_nyc[['latitude', 'longitude']]
locationlist = locations.values.tolist()
l = len(locationlist)
base_map = folium.Map(location=defeault_location)
marker_cluster = MarkerCluster(locationlist)
marker_cluster.add_to(base_map)
base_map.save('k-means.html')

# calculate distortion for a range of number of cluster
# distortions = []
# for i in range(1, 20):
#     km = KMeans(
#         n_clusters=i, init='random',
#         n_init=10, max_iter=300,
#         tol=1e-04, random_state=0
#     )
#     km.fit(df)
#     distortions.append(km.inertia_)
#
# # plot
# plt.plot(range(1, 20), distortions, marker='o')
# plt.xlabel('Number of clusters')
# plt.ylabel('Distortion')
# plt.show()