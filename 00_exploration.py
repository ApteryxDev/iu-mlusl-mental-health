# =============================================================================
# 00_exploration.py  --  Step 1 of 4: getting to know the data
# =============================================================================
# Course: DLBDSMLUSL01, Case Study Task 1 (Mental Health in Tech, OSMI 2016).
# This script goes with chapter 2 of my report ("Exploring the Data").
#
# The task tips say to start by exploring the data with some descriptive
# statistics and a few plots before doing anything else. That is what this
# file is for. I am NOT cleaning or modelling here yet -- I just want to
# understand what I am dealing with, because the survey is big (63 questions)
# and I have been warned it is messy (missing values, free text).
#
# By the end of this script I want answers to four questions:
#   - How big is the data and what types are the columns?
#   - How bad is the missing-value problem, and is it random or not?
#   - What is wrong with the gender and age columns?
#   - Which columns are free text that I will not be able to use directly?
#
# Everything I learn here is what I use to plan the cleaning in script 01.
# =============================================================================

import pandas as pd
import matplotlib
matplotlib.use("Agg")          # I save plots to files instead of opening windows
import matplotlib.pyplot as plt

# The CSV is inside a sub-folder called task1_data. Note: the file name has a
# typo ("heath" instead of "health") -- that typo is in the original Kaggle
# file, so I keep it exactly as is, otherwise pandas will not find the file.
RAW = "task1_data/mental-heath-in-tech-2016_20161114.csv"
df = pd.read_csv(RAW)

# -----------------------------------------------------------------------------
# Question 1: how big is the data, and what kind of columns are there?
# -----------------------------------------------------------------------------
# I expect mostly text answers ("Yes", "No", "Maybe"), so let me confirm that.
print("Rows:", df.shape[0], "Columns:", df.shape[1])
print("Column types:", df.dtypes.value_counts().to_dict())
# If most columns are "object", that means text -> I will need to turn them
# into numbers later, because the clustering algorithms only take numbers.

# -----------------------------------------------------------------------------
# Question 2: how many missing values, and are they random?
# -----------------------------------------------------------------------------
# .isna() marks every empty cell as True; .mean() then gives the fraction of
# empty cells per column (because True counts as 1). I sort to see the worst.
missing_fraction = df.isna().mean().sort_values(ascending=False)

print("\nColumns with NO missing values:",
      (missing_fraction == 0).sum(), "out of", len(missing_fraction))
print("The 5 emptiest columns:")
for column_name in missing_fraction.head(5).index:
    print("  ", round(missing_fraction[column_name] * 100), "% empty -", column_name[:65])

# This next check is the most important thing in the whole exploration.
# I noticed many empty cells are in "employer" questions. My guess: maybe the
# self-employed people were simply never asked those questions. If that is
# true, then the number of self-employed people should EXACTLY equal the number
# of blanks in an employer question. Let me test that guess directly.
self_employed_count = (df["Are you self-employed?"] == 1).sum()
employer_question = ("Does your employer provide mental health benefits "
                     "as part of healthcare coverage?")
blanks_in_employer_question = df[employer_question].isna().sum()

print("\nSelf-employed people:", self_employed_count)
print("Blanks in an employer question:", blanks_in_employer_question)
if self_employed_count == blanks_in_employer_question:
    print("-> The two numbers match exactly. So the blanks are NOT random:")
    print("   the survey just never showed these questions to the self-employed.")
    print("   This means I must NOT fill these blanks with a guessed value;")
    print("   I will treat 'not asked' as its own category later.")

# A picture makes the missingness easy to see in the report. I plot the 20
# emptiest columns as horizontal bars. I reverse the order (iloc[::-1]) so the
# emptiest one ends up at the top of the chart.
top20 = missing_fraction.head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(range(len(top20)), top20.values, color="steelblue")
ax.set_yticks(range(len(top20)))
ax.set_yticklabels([name[:60] for name in top20.index], fontsize=7)
ax.set_title("20 most incomplete columns (fraction missing)")
ax.set_xlabel("fraction missing")
plt.tight_layout()
plt.savefig("eda_missingness.png", dpi=130)
plt.close()

# -----------------------------------------------------------------------------
# Question 3: what is wrong with the age and gender columns?
# -----------------------------------------------------------------------------
# I was told gender is free text. Let me see how many different spellings there
# are -- if it is a lot, I will have to standardise it before I can use it.
print("\nNumber of different gender spellings:", df["What is your gender?"].nunique())

# Age should be a normal adult range. Let me check the min and max for errors.
print("Age min:", df["What is your age?"].min(),
      " Age max:", df["What is your age?"].max())
# (If I see something like 3 or 329 here, those are obviously typing mistakes
#  that I will need to fix in the cleaning step.)

# Plot both so I can show them in the report.
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
df["What is your age?"].clip(0, 100).hist(bins=40, ax=ax[0], color="steelblue")
ax[0].set_title("Age distribution (watch the outliers)")
ax[0].set_xlabel("age")
df["What is your gender?"].value_counts().head(10).plot(kind="bar", ax=ax[1], color="indianred")
ax[1].set_title("Top 10 raw gender answers (out of many)")
ax[1].tick_params(labelsize=7)
plt.tight_layout()
plt.savefig("eda_age_gender.png", dpi=130)
plt.close()

# -----------------------------------------------------------------------------
# Question 4: which columns are free text I cannot encode?
# -----------------------------------------------------------------------------
# If a column has almost as many unique answers as there are people, it is
# basically free text (everyone wrote something different) and one-hot encoding
# it would create thousands of useless columns. Let me check the two opinion
# questions and the job-position question.
for column_name in ["Why or why not?",
                    "Which of the following best describes your work position?"]:
    print("Unique answers in '" + column_name[:40] + "...':",
          df[column_name].nunique())

print("\nDone exploring. Saved eda_missingness.png and eda_age_gender.png.")
print("Next: use these findings to clean the data in 01_preprocessing.py")
