# =============================================================================
# 03_refined_iteration.py  --  Step 4 of 4: stepping back and doing it properly
# =============================================================================
# Course: DLBDSMLUSL01, Case Study Task 1. Goes with chapters 6-8 of my report.
#
# Script 02 taught me that my first attempt just rediscovered the survey
# structure (self-employed vs not). The task tips encourage exactly this:
# "Building from this, you can elaborate and take some steps back to improve
# the quality of your work." So here are my two fixes:
#
#   Fix 1 - only keep employed people. HR's programme is about the workplace,
#           and the self-employed never answered the workplace questions, so
#           keeping them just adds the noise that ruined iteration 1.
#
#   Fix 2 - only cluster on a small set of ATTITUDE questions (how comfortable
#           people are talking about mental health, whether they fear
#           consequences, whether they feel supported, treatment history),
#           instead of the whole encoded table full of "Not applicable" columns.
#
# Then I choose the number of clusters carefully (four different checks), and I
# compare k-Means (which forces everyone into one group) with a Gaussian
# Mixture Model / GMM (which is softer and gives each person a probability of
# belonging to each group). Finally I describe the clusters so HR can act.
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

# I load the ORIGINAL survey again here (not the encoded table) because for the
# attitude questions I want the real text answers ("Yes"/"No"/"Maybe"), which
# are easier to read and to describe at the end. I also load the demographics.
survey = pd.read_csv("task1_data/mental-heath-in-tech-2016_20161114.csv")
survey = survey.reset_index(drop=True)
profile = pd.read_csv("profiling.csv")

# -----------------------------------------------------------------------------
# Fix 1: keep only the employed respondents.
# -----------------------------------------------------------------------------
# "Are you self-employed?" is 0 for employed people. I use that to filter both
# the survey and the matching demographics table so the rows stay aligned.
is_employed = survey["Are you self-employed?"] == 0
survey_employed = survey[is_employed].reset_index(drop=True)
profile = profile[is_employed.values].reset_index(drop=True)
print("Employed respondents kept:", len(survey_employed))

# -----------------------------------------------------------------------------
# Fix 2: choose a small set of attitude questions.
# -----------------------------------------------------------------------------
# I do not want all 60 questions -- only the ones about feelings and workplace
# experience. The question texts are long, so instead of typing each full
# title I list a few key phrases and keep any column whose title contains one
# of them. I spell this out as a loop so it is easy to follow which questions
# get included and why some are skipped.
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
    # skip the open-text answers and the "if yes..." follow-ups (free text)
    if column.startswith(("Why", "If yes", "If so", "If maybe")):
        continue
    # skip "previous employer" questions: in iteration testing these created
    # their own split (people at their first job answered them all as blank),
    # which is the same structural problem I am trying to avoid.
    if "previous" in column.lower():
        continue
    # keep the column if its title mentions any phrase from my list
    for phrase in phrases_i_want:
        if phrase in column:
            attitude_questions.append(column)
            break
print("Number of attitude questions used:", len(attitude_questions))

# Build a small table from just those questions. Any remaining blank I label
# "No answer" so it becomes a normal category rather than a hole.
subset = survey_employed[attitude_questions].copy()
subset = subset.fillna("No answer")

# Turn the text answers into 0/1 columns (one column per possible answer).
X = pd.get_dummies(subset)
X = X.astype(float)

# Remove any column that is always the same -- it carries no information and
# would just slow PCA down without helping.
columns_with_variation = []
for column in X.columns:
    if X[column].var() > 0:
        columns_with_variation.append(column)
X = X[columns_with_variation]

# Scale and run PCA, keeping ~80% of the variance. This is the same counting
# loop I used in script 02 (kept identical on purpose, for consistency).
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
# Choosing how many clusters (k).
# -----------------------------------------------------------------------------
# There is no single "correct" k, so I look at four different measures for
# k from 2 to 8 and then make a reasoned choice:
#   - inertia (elbow): how tight the clusters are; I look for the "bend".
#   - silhouette: how well separated they are (higher = better).
#   - Davies-Bouldin: how much clusters overlap (lower = better).
#   - BIC (for the GMM): balances fit against complexity (lower = better).
k_values = list(range(2, 9))
elbow_scores = []
silhouette_scores = []
davies_bouldin_scores = []
bic_scores = []

for k in k_values:
    # k-Means side
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=RANDOM)
    kmeans_labels = kmeans.fit_predict(Z)
    elbow_scores.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(Z, kmeans_labels))
    davies_bouldin_scores.append(davies_bouldin_score(Z, kmeans_labels))
    # GMM side
    gmm = GaussianMixture(n_components=k, covariance_type="diag",
                          random_state=RANDOM, n_init=3)
    gmm.fit(Z)
    bic_scores.append(gmm.bic(Z))

# Print the four measures as a small table so I can compare them in the report.
print("\n k | inertia | silhouette | DaviesBouldin | BIC")
for i in range(len(k_values)):
    print(" {} | {:7.0f} | {:10.3f} | {:13.2f} | {:.0f}".format(
        k_values[i], elbow_scores[i], silhouette_scores[i],
        davies_bouldin_scores[i], bic_scores[i]))

# My decision: the silhouette technically likes k=2, but two groups barely
# tell HR anything. At k=3 the groups are a good size and each tells a clear,
# different story, so I go with 3. I explain this trade-off in chapter 6.1.
# (Picking interpretability over a metric is a judgement call, and I say so.)
chosen_k = 3

# -----------------------------------------------------------------------------
# Fit the two chosen models: k-Means (hard) and GMM (soft).
# -----------------------------------------------------------------------------
kmeans = KMeans(n_clusters=chosen_k, n_init=10, random_state=RANDOM)
kmeans_labels = kmeans.fit_predict(Z)

gmm = GaussianMixture(n_components=chosen_k, covariance_type="diag",
                      random_state=RANDOM, n_init=5)
gmm.fit(Z)
gmm_labels = gmm.predict(Z)
gmm_probabilities = gmm.predict_proba(Z)   # probability of each group, per person

# How often do the two methods agree? I line up each k-Means cluster with the
# GMM cluster it overlaps with most (using a cross-tab table of counts), then
# add up the overlaps and divide by the number of people.
overlap_table = pd.crosstab(kmeans_labels, gmm_labels)
best_overlap_per_cluster = overlap_table.values.max(axis=1)
agreement = best_overlap_per_cluster.sum() / len(Z)

# How sure is the GMM about each person? predict_proba gives a probability for
# each group; the biggest of those is its confidence. If that biggest value is
# under 0.6, the GMM is basically undecided about that person -- they sit
# between groups. I count how common that is.
highest_probability = gmm_probabilities.max(axis=1)
uncertain_share = (highest_probability < 0.6).mean()

print("\nChosen number of clusters:", chosen_k)
print("k-Means and the GMM agree on", round(agreement * 100), "% of people.")
print("The GMM is undecided (under 60% sure) about",
      round(uncertain_share * 100), "% of people -- these are the in-between cases.")

# Attach the k-Means cluster number to each person for the profiling below.
profile["cluster"] = kmeans_labels
survey_employed["cluster"] = kmeans_labels

# -----------------------------------------------------------------------------
# Figure 1: the four model-selection plots side by side.
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
# Figure 2: the clusters drawn on the first two PCA components.
# -----------------------------------------------------------------------------
# Unlike iteration 1 (four separate corners), I expect one connected cloud that
# grades from "closed" on one side to "open" on the other, which is why the
# soft GMM assignment makes sense here.
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
# Describe each cluster -- this is the part that actually answers HR.
# -----------------------------------------------------------------------------
# For a few key questions I show, per cluster, the percentage giving each
# answer. Reading these percentages is how I name the groups in the report
# (e.g. one cluster is 84% "not comfortable" with their supervisor).
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
    # normalize="index" turns the counts into percentages within each cluster
    answers_by_cluster = pd.crosstab(
        survey_employed["cluster"], survey_employed[question],
        normalize="index") * 100
    report_file.write("=== " + question + " ===\n")
    report_file.write(answers_by_cluster.round(0).to_string() + "\n\n")
report_file.close()

print("\nSaved the two figures and cluster_profiles_employed.txt")
print("These cluster descriptions are what I turn into HR recommendations.")
