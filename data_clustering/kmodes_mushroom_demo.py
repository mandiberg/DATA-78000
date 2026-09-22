import warnings

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from kmodes.kmodes import KModes

warnings.filterwarnings(
    "ignore",
    message=r"The resize_event function was deprecated.*",
    category=matplotlib.MatplotlibDeprecationWarning,
)

'''
This code pattern demonstrates the K-Modes clustering algorithm on the standard mushroom dataset.
It also uses the Elbow Method to determine the optimal number of clusters.
Distance is calculated using Hamming distance, which is a way to quantify the difference between categorical data.
Always remember that you should set your k based on your domain knowledge and your knowldge of your data
The Elbow Method is just a heuristic, not a definitive answer. 
Use your critical thinking skills!!!
'''

# 1. Fetching the UCI Mushroom Dataset via URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
# The original dataset has columns without headers; column 0 is the label (edible/poisonous)
columns = [
    "target",
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat",
]
df = pd.read_csv(url, header=None, names=columns)

# Drop target label for unsupervised learning, and compress row count for fast classroom execution
X = df.drop(columns=["target"]).head(1000)

# 2. Calculate the cost for the Elbow Method (k from 1 to 5)
cost = []
k_range = range(1, 6)

for k in k_range:
    # 'Cao' initialization is highly recommended over random for K-Modes
    km = KModes(n_clusters=k, init="Cao", n_init=1, verbose=0)
    km.fit(X)
    cost.append(km.cost_)  # Cost = Total number of feature mismatches (Hamming distance)

# 3. Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(k_range, cost, marker="o", linestyle="--", color="purple")
plt.title("K-Modes Elbow Method (Mushroom Data)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Clustering Cost (Total Mismatches)")
plt.xticks(k_range)
plt.grid(axis="y", linestyle=":", alpha=0.6)
plt.show()

# 4. Fit the Final Model (Choosing k=2 based on our binary domain expectation)
final_km = KModes(n_clusters=2, init="Cao", n_init=1, verbose=0)
clusters = final_km.fit_predict(X)

# 5. Display the calculated Centroids (Modes)
print("\n--- Discovered Cluster Modes (Most Frequent Categories) ---")
centroids_df = pd.DataFrame(final_km.cluster_centroids_, columns=X.columns)
print(centroids_df[["cap-shape", "cap-color", "odor", "habitat"]])
