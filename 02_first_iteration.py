# =============================================================================
# 02_first_iteration.py  --  Step 3 of 4: my first, simple attempt
# =============================================================================
# Course: DLBDSMLUSL01, Case Study Task 1. Goes with chapter 5 of my report
# ("Iteration 1: Simple and Quick -- and Wrong").
#
# The task tips literally say: "keep it simple in the first iteration and try
# to come up with quick solutions". So I am not being clever here. I just take
# the clean table from script 01, shrink it with PCA, and run k-Means. The
# idea is to get SOMETHING working, look at it, and then improve.
#
# Spoiler for my own notes: this attempt does not really work, but it fails in
# a useful way. It teaches me that the survey's structure (who was asked what)
# is louder in the data than people's actual attitudes. I keep this script in
# the final hand-in because explaining this failure is part of the story, and
# the task asks me to "critically assess" my decisions.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# I fix the random seed to 42 so that every time I run the script I get the
# exact same numbers. Without this, k-Means could start from different random
# points and give slightly different clusters each run, which would be
# confusing when I write about specific numbers in the report.
RANDOM = 42

# Load the clean data and the demographics table from script 01.
X = pd.read_csv("X_features.csv")
profile = pd.read_csv("profiling.csv")

# -----------------------------------------------------------------------------
# Step 1: scaling.
# -----------------------------------------------------------------------------
# PCA looks at which directions vary the most. If one column happened to be on
# a bigger numeric scale than the others, it would dominate just because of its
# scale, not because it is more important. StandardScaler fixes this by putting
# every column on the same footing (mean 0, similar spread) before PCA.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------------------------------------------
# Step 2: PCA to reduce the ~170 columns down to a handful.
# -----------------------------------------------------------------------------
# First I fit PCA on everything just to read off how much variance each new
# component explains. Then I decide how many components I actually need.
pca = PCA(random_state=RANDOM)
pca.fit(X_scaled)

# I want enough components to keep about 80% of the information (variance).
# Rather than use a fancy one-liner, I just add up the percentages one by one
# until the running total passes 0.80, and count how many that took.
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
# That last number being big is already a little suspicious -- one single
# direction explaining a lot usually means one feature is splitting the data.

# Now redo PCA keeping only that many components, and transform the data.
pca = PCA(n_components=n_components, random_state=RANDOM)
Z = pca.fit_transform(X_scaled)

# -----------------------------------------------------------------------------
# Step 3: investigate what that strong first component actually is.
# -----------------------------------------------------------------------------
# My hunch: because I encoded "not asked" as its own category in script 01,
# the self-employed people (who had loads of "not asked" answers) might sit far
# apart from everyone else. If component 1 is really just "self-employed yes/no"
# then it will line up almost perfectly with the self_employed column.
# I check this with a correlation: +1 or -1 means they move together perfectly.
correlation = np.corrcoef(Z[:, 0], profile["self_employed"])[0, 1]
print("Correlation between component 1 and 'self-employed':",
      round(correlation, 2), " <-- if this is near 1, that is my problem")

# -----------------------------------------------------------------------------
# Step 4: a quick k-Means with 3 clusters, no tuning at all.
# -----------------------------------------------------------------------------
# I just pick k=3 to get a first look. I am not justifying the number yet --
# that careful choice happens in the next script.
kmeans = KMeans(n_clusters=3, n_init=10, random_state=RANDOM)
labels = kmeans.fit_predict(Z)

# Now I look at each cluster and ask: what fraction of it is self-employed?
# If the clusters are really about attitudes, self-employment should be mixed
# through them. If instead each cluster is basically all-or-nothing on
# self-employment, then the clustering is just repeating the survey structure.
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
# Step 5: a picture showing the problem.
# -----------------------------------------------------------------------------
# If component 1 really is self-employment, then colouring points by
# self-employment should match the left-right split in the plot. That visual
# is much more convincing in the report than just stating the correlation.
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
