# =============================================================================
# 03_refined_iteration.py -- step 4 of 4: stepping back and doing it properly
# =============================================================================
# DLBDSMLUSL01, Task 1. Goes with chapters 6-8 of the report.
#
# Script 02 just rediscovered the survey structure (self-employed vs not). Two
# fixes here:
#   Fix 1 - keep only employed people. HR's programme is about the workplace,
#           and the self-employed never answered the workplace questions, so
#           they only add the noise that wrecked iteration 1.
#   Fix 2 - cluster only on a small set of ATTITUDE questions (comfort talking
#           about mental health, fear of consequences, feeling supported,
#           treatment history) instead of the whole encoded table.
# Then choose k carefully (four checks) and compare k-Means (hard) with a GMM
# (soft, gives each person a probability per group). Finally describe the
# clusters so HR can act on them.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score

RANDOM = 42

# load the ORIGINAL survey, not the encoded table -- for the attitude questions
# I want the real "Yes"/"No"/"Maybe" text, which is easier to describe at the end
survey = pd.read_csv("task1_data/mental-heath-in-tech-2016_20161114.csv")
survey = survey.reset_index(drop=True)
profile = pd.read_csv("profiling.csv")

# -----------------------------------------------------------------------------
# Fix 1: keep only the employed respondents.
# -----------------------------------------------------------------------------
# self-employed == 0 means employed. filter both the survey and the demographics
# table with the same mask so the rows stay aligned.
is_employed = survey["Are you self-employed?"] == 0
survey_employed = survey[is_employed].reset_index(drop=True)
profile = profile[is_employed.values].reset_index(drop=True)
print("Employed respondents kept:", len(survey_employed))

# -----------------------------------------------------------------------------
# Fix 2: pick a small set of attitude questions.
# -----------------------------------------------------------------------------
# not all 60 questions -- only feelings/workplace ones. The titles are long, so
# I match on key phrases instead of typing each full title.
phrases_i_want = [
    "comfortable discussing",
    "negative consequences",
    "takes mental health",
    "currently have a mental health",
    "sought treatment",
    "provide mental health benefits",
    "options for mental health care",
    "willing to discuss",
    "bring up a mental health",
    "observed negative consequences",
    "supportive of mental",
]

attitude_questions = []
for column in survey_employed.columns:
    # skip open text and the "if yes..." follow-ups
    if column.startswith(("Why", "If yes", "If so", "If maybe")):
        continue
    # skip "previous employer" questions -- in testing these made their own split
    # (first-jobbers answered them all blank), the same structural trap as before
    if "previous" in column.lower():
        continue
    for phrase in phrases_i_want:
        if phrase in column:
            attitude_questions.append(column)
            break
print("Number of attitude questions used:", len(attitude_questions))

# small table from just those questions; remaining blanks -> "No answer"
subset = survey_employed[attitude_questions].copy()
subset = subset.fillna("No answer")

X = pd.get_dummies(subset)
X = X.astype(float)

# drop constant columns (no information, just slows PCA)
columns_with_variation = []
for column in X.columns:
    if X[column].var() > 0:
        columns_with_variation.append(column)
X = X[columns_with_variation]

# same scale + PCA-to-80% as script 02
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(random_state=RANDOM)
pca.fit(X_scaled)
variance_so_far = 0.0
n_components = 0
for ratio in pca.explained_variance_ratio_:
    variance_so_far = variance_so_far + ratio
    n_components = n_components + 1
    if variance_so_far >= 0.80:
        break
pca = PCA(n_components=n_components, random_state=RANDOM)
Z = pca.fit_transform(X_scaled)
print("Feature table:", X.shape[0], "rows,", X.shape[1], "columns.")
print("PCA keeps", n_components, "components for 80% of the variance.")

# -----------------------------------------------------------------------------
# How many clusters (k)?
# -----------------------------------------------------------------------------
# no single right answer, so look at four measures for k = 2..8:
#   inertia (elbow)  -> tightness, look for the bend
#   silhouette       -> separation, higher better
#   Davies-Bouldin   -> overlap, lower better
#   BIC (GMM)        -> fit vs complexity, lower better
k_values = list(range(2, 9))
elbow_scores = []
silhouette_scores = []
davies_bouldin_scores = []
bic_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=RANDOM)
    kmeans_labels = kmeans.fit_predict(Z)
    elbow_scores.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(Z, kmeans_labels))
    davies_bouldin_scores.append(davies_bouldin_score(Z, kmeans_labels))
    gmm = GaussianMixture(n_components=k, covariance_type="diag",
                          random_state=RANDOM, n_init=3)
    gmm.fit(Z)
    bic_scores.append(gmm.bic(Z))

print("\n k | inertia | silhouette | DaviesBouldin | BIC")
for i in range(len(k_values)):
    print(" {} | {:7.0f} | {:10.3f} | {:13.2f} | {:.0f}".format(
        k_values[i], elbow_scores[i], silhouette_scores[i],
        davies_bouldin_scores[i], bic_scores[i]))

# silhouette technically prefers k=2, but two groups tell HR almost nothing. at
# k=3 the groups are a decent size and each tells a different story, so I go with
# 3 and explain the trade-off in 6.1. picking interpretability over a metric is a
# judgement call -- I say so rather than pretending the metric chose for me.
chosen_k = 3

# -----------------------------------------------------------------------------
# Fit both models: k-Means (hard) and GMM (soft).
# -----------------------------------------------------------------------------
kmeans = KMeans(n_clusters=chosen_k, n_init=10, random_state=RANDOM)
kmeans_labels = kmeans.fit_predict(Z)

gmm = GaussianMixture(n_components=chosen_k, covariance_type="diag",
                      random_state=RANDOM, n_init=5)
gmm.fit(Z)
gmm_labels = gmm.predict(Z)
gmm_probabilities = gmm.predict_proba(Z)   # probability of each group, per person

# how often do the two agree? match each k-Means cluster to the GMM cluster it
# overlaps with most (cross-tab), sum the overlaps, divide by n.
overlap_table = pd.crosstab(kmeans_labels, gmm_labels)
best_overlap_per_cluster = overlap_table.values.max(axis=1)
agreement = best_overlap_per_cluster.sum() / len(Z)

# how sure is the GMM per person? its biggest probability is its confidence.
# under 0.6 = basically undecided, person sits between groups. count those.
highest_probability = gmm_probabilities.max(axis=1)
uncertain_share = (highest_probability < 0.6).mean()

print("\nChosen number of clusters:", chosen_k)
print("k-Means and the GMM agree on", round(agreement * 100), "% of people.")
print("The GMM is undecided (under 60% sure) about",
      round(uncertain_share * 100), "% of people -- these are the in-between cases.")

profile["cluster"] = kmeans_labels
survey_employed["cluster"] = kmeans_labels

# -----------------------------------------------------------------------------
# Figure 1: the four model-selection plots.
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
axes[0].plot(k_values, elbow_scores, marker="o")
axes[0].set_title("Elbow (inertia)")
axes[1].plot(k_values, silhouette_scores, marker="o")
axes[1].set_title("Silhouette (higher is better)")
axes[2].plot(k_values, davies_bouldin_scores, marker="o")
axes[2].set_title("Davies-Bouldin (lower is better)")
axes[3].plot(k_values, bic_scores, marker="o")
axes[3].set_title("GMM BIC (lower is better)")
for ax in axes:
    ax.set_xlabel("number of clusters k")
plt.tight_layout()
plt.savefig("fig_iter2_selection.png", dpi=130)
plt.close()

# -----------------------------------------------------------------------------
# Figure 2: clusters on the first two PCA components.
# -----------------------------------------------------------------------------
# unlike iteration 1 (separate corners) I expect one connected cloud grading
# from "closed" to "open" -- which is why the soft GMM assignment fits here.
plt.figure(figsize=(6.5, 5))
plt.scatter(Z[:, 0], Z[:, 1], c=kmeans_labels, cmap="Set2", s=14, alpha=0.75)
plt.title("Iteration 2: three attitude clusters (employed people only)")
plt.xlabel("Component 1 (open vs. closed)")
plt.ylabel("Component 2")
plt.colorbar(label="cluster")
plt.tight_layout()
plt.savefig("fig_iter2_clusters.png", dpi=130)
plt.close()

# -----------------------------------------------------------------------------
# Describe each cluster -- the part that actually answers HR.
# -----------------------------------------------------------------------------
# per cluster, the % giving each answer to a few key questions. reading these is
# how I name the groups in the report (e.g. one is 84% "not comfortable").
key_questions = [
    "Do you currently have a mental health disorder?",
    "Have you ever sought treatment for a mental health issue from a mental health professional?",
    "Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?",
    "Do you think that discussing a mental health disorder with your employer would have negative consequences?",
    "Do you feel that your employer takes mental health as seriously as physical health?",
]

report_file = open("cluster_profiles_employed.txt", "w")
cluster_sizes = []
for cluster_id in range(chosen_k):
    cluster_sizes.append(int((kmeans_labels == cluster_id).sum()))
report_file.write("Number of clusters: " + str(chosen_k) + "\n")
report_file.write("Cluster sizes: " + str(cluster_sizes) + "\n\n")

for question in key_questions:
    # normalize="index" -> counts become percentages within each cluster
    answers_by_cluster = pd.crosstab(
        survey_employed["cluster"], survey_employed[question],
        normalize="index") * 100
    report_file.write("=== " + question + " ===\n")
    report_file.write(answers_by_cluster.round(0).to_string() + "\n\n")
report_file.close()

print("\nSaved the two figures and cluster_profiles_employed.txt")
print("These cluster descriptions are what I turn into HR recommendations.")
