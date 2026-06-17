# =============================================================================
# 02_first_iteration.py step 3 of 4: first, simple attempt
# =============================================================================
# DLBDSMLUSL01, Task 1. Goes with chapter 5 of the report.
#
# The task tips say to keep the first iteration simple and quick, so I'm not
# being clever: take the clean table from script 01, shrink it with PCA, run
# k-Means, look at it. This one ends up not really working but it's worth
# keeping in the hand-in, because the WAY it fails (the survey structure being
# louder than people's attitudes) is what motivates iteration 2, and the task
# asks me to critically assess my own decisions.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# fixed seed so the numbers I quote in the report are reproducible (k-Means
# starts from random points otherwise and the clusters shift slightly each run)
RANDOM = 42

X = pd.read_csv("X_features.csv")
profile = pd.read_csv("profiling.csv")

# -----------------------------------------------------------------------------
# Step 1: scaling.
# -----------------------------------------------------------------------------
# PCA picks the directions of biggest variance, so a column on a larger numeric
# scale would dominate just for being bigger. StandardScaler puts everything on
# the same footing first.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------------------------------------------
# Step 2: PCA down from ~170 columns to a handful.
# -----------------------------------------------------------------------------
# fit on everything first just to read off the variance per component, then
# decide how many to keep.
pca = PCA(random_state=RANDOM)
pca.fit(X_scaled)

# keep enough components for ~80% of the variance: add the ratios up until the
# running total passes 0.80 and count how many that took.
variance_so_far = 0.0
n_components = 0
for ratio in pca.explained_variance_ratio_:
    variance_so_far = variance_so_far + ratio
    n_components = n_components + 1
    if variance_so_far >= 0.80:
        break
print("PCA needs", n_components, "components to keep 80% of the variance.")
print("Just the first component already explains",
      round(pca.explained_variance_ratio_[0] * 100, 1), "% of the variance.")
# one component explaining that much is suspicious -- usually means a single
# feature is splitting the data

pca = PCA(n_components=n_components, random_state=RANDOM)
Z = pca.fit_transform(X_scaled)

# -----------------------------------------------------------------------------
# Step 3: what IS that strong first component?
# -----------------------------------------------------------------------------
# guess: encoding "not asked" as its own category in script 01 pushed the
# self-employed (lots of "not asked" answers) far from everyone else. If
# component 1 is basically "self-employed yes/no", it'll correlate ~1 with the
# self_employed column. check:
correlation = np.corrcoef(Z[:, 0], profile["self_employed"])[0, 1]
print("Correlation between component 1 and 'self-employed':",
      round(correlation, 2), " <-- if this is near 1, that is my problem")

# -----------------------------------------------------------------------------
# Step 4: quick k-Means, k=3, no tuning.
# -----------------------------------------------------------------------------
# k=3 is just a first look -- the proper choice of k is in script 03.
kmeans = KMeans(n_clusters=3, n_init=10, random_state=RANDOM)
labels = kmeans.fit_predict(Z)

# per cluster: what fraction is self-employed? if attitudes were driving this,
# self-employment would be mixed through the clusters. if instead each cluster
# is all-or-nothing on it, the clustering is just echoing the survey structure.
print("\nWhat each cluster is made of:")
for cluster_id in [0, 1, 2]:
    people = profile[labels == cluster_id]
    size = len(people)
    share = people["self_employed"].mean() * 100
    print("  cluster", cluster_id, ":", size, "people,", round(share), "% self-employed")

print("\nConclusion: the clusters split mainly by employment status")
print("(employed / self-employed / first job), not by how people feel about")
print("mental health. The data prep was fine, but I clustered on the wrong")
print("things. I will fix this in 03_refined_iteration.py.")

# -----------------------------------------------------------------------------
# Step 5: a picture of the problem.
# -----------------------------------------------------------------------------
# colour the points by self-employment: if it lines up with the left-right
# split, component 1 really is self-employment. more convincing in the report
# than just stating the correlation.
plt.figure(figsize=(6.5, 5))
plt.scatter(Z[:, 0], Z[:, 1], c=profile["self_employed"],
            cmap="coolwarm", s=10, alpha=0.6)
plt.title("Iteration 1: PCA of the full data, coloured by self-employment")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.colorbar(label="self-employed (1 = yes)")
plt.tight_layout()
plt.savefig("fig_iter1_pca.png", dpi=130)
plt.close()
print("Saved fig_iter1_pca.png")
