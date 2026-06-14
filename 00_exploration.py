# =============================================================================
# 00_exploration.py -- step 1 of 4: looking at the data before touching anything
# =============================================================================
# DLBDSMLUSL01, Task 1 (OSMI 2016 Mental Health in Tech). Goes with chapter 2
# of the report. No cleaning or modelling here yet -- the survey is big (63
# questions) and messy, so I just want to know what I'm dealing with first:
# how big it is, how bad the missing values are, what's wrong with age/gender,
# and which columns are free text. Whatever I find here decides the cleaning
# plan in script 01.
# =============================================================================

import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Agg = write PNGs instead of opening windows
import matplotlib.pyplot as plt

# the filename typo "heath" is in the original Kaggle file -- leave it, or
# pandas won't find it
RAW = "task1_data/mental-heath-in-tech-2016_20161114.csv"
df = pd.read_csv(RAW)

# print(df.columns.tolist())   # used this to copy the exact question titles

# -----------------------------------------------------------------------------
# how big is it, and what column types?
# -----------------------------------------------------------------------------
print("Rows:", df.shape[0], "Columns:", df.shape[1])
print("Column types:", df.dtypes.value_counts().to_dict())
# mostly "object" = text, which I'll have to turn into numbers later

# -----------------------------------------------------------------------------
# missing values -- and are they random?
# -----------------------------------------------------------------------------
missing_fraction = df.isna().mean().sort_values(ascending=False)

print("\nColumns with NO missing values:",
      (missing_fraction == 0).sum(), "out of", len(missing_fraction))
print("The 5 emptiest columns:")
for column_name in missing_fraction.head(5).index:
    print("  ", round(missing_fraction[column_name] * 100), "% empty -", column_name[:65])

# A lot of the empty cells are in "employer" questions. Hunch: the self-employed
# were just never asked those. If that's right, the count of self-employed people
# should match the number of blanks in an employer question exactly. Test it:
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

# bar chart of the 20 emptiest columns for the report. iloc[::-1] so the
# emptiest ends up at the top of the chart, not the bottom.
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
# age and gender
# -----------------------------------------------------------------------------
# gender is free text -- how many spellings am I actually dealing with?
print("\nNumber of different gender spellings:", df["What is your gender?"].nunique())

print("Age min:", df["What is your age?"].min(),
      " Age max:", df["What is your age?"].max())
# values way outside a normal adult range are typing mistakes -> fix in script 01

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
# which columns are free text I can't encode?
# -----------------------------------------------------------------------------
# if a column has nearly as many unique answers as there are rows, everyone
# wrote something different -> one-hot would explode into thousands of columns
for column_name in ["Why or why not?",
                    "Which of the following best describes your work position?"]:
    print("Unique answers in '" + column_name[:40] + "...':",
          df[column_name].nunique())

print("\nDone exploring. Saved eda_missingness.png and eda_age_gender.png.")
print("Next: use these findings to clean the data in 01_preprocessing.py")
