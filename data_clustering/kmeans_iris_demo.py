
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris


'''
This code pattern demonstrates the K-Means clustering algorithm on the standard Iris dataset.
It uses the Elbow Method to determine the optimal number of clusters.
Distance is calculated using Euclidean distance, which is fancy way of saying the straight-line distance between two points in space... just in hyperdimensional space(!)
Always remember that you should set your k based on your domain knowledge and your knowldge of your data
The Elbow Method is just a heuristic, not a definitive answer. 
Use your critical thinking skills!!!
'''

# Set your k value
# For this dataset, we will learn that the correct k is 3
k = 3

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Using all 4 features: sepal/petal length and width

# Calculate inertia for the Elbow Method (k from 1 to 10)
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# visualize results side-by-side 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# do elbow plot for k selection
ax1.plot(k_range, inertia, marker="o", linestyle="--", color="b")
ax1.set_title("The Elbow Method")
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
ax1.set_xticks(k_range)

# fit the actual model, using k=3
final_kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
y_kmeans = final_kmeans.fit_predict(X)

# Plot 2: Final K-Means Clusters
# For visualization, we plot the first two features (Sepal Length vs Sepal Width)
ax2.scatter(
    X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap="viridis", alpha=0.8, label="Data"
)

# plot the centroids (this is an approximation using only the first two dimensions of the 4D centroids)
centroids = final_kmeans.cluster_centers_
ax2.scatter(
    centroids[:, 0],
    centroids[:, 1],
    c="red",
    s=200,
    marker="X",
    label="Centroids",
)

ax2.set_title("K-Means Clustering on Iris (k=3)")
ax2.set_xlabel(iris.feature_names[0])  # Sepal length
ax2.set_ylabel(iris.feature_names[1])  # Sepal width
ax2.legend()

plt.tight_layout()
plt.show()

# print out the final cluster data
print("Final K-Means Clustering Results:")
for i in range(k):
    cluster_points = X[y_kmeans == i]
    print(f"\nCluster {i + 1}:")
    print(cluster_points)
    print(f"Type of first point in cluster {i + 1}: {type(cluster_points[0])}")
