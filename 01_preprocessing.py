# =============================================================================
# 01_preprocessing.py -- step 2 of 4: cleaning, and turning text into numbers
# =============================================================================
# DLBDSMLUSL01, Task 1. Goes with chapter 4 of the report.
#
# Script 00 found four problems: structural missing values, a messy free-text
# gender column, impossible ages, and free-text opinion columns. This script
# fixes all of them and produces a numeric table the sklearn libraries can use.
# I keep a running `log` and print every decision so I can quote the exact
# numbers (how many columns each step touched) in the report.
# =============================================================================

import re
import numpy as np
import pandas as pd

RAW = "task1_data/mental-heath-in-tech-2016_20161114.csv"  # keep the "heath" typo
log = []

def note(message):
    # print it AND keep it, so I can dump the whole list to a file at the end
    print(message)
    log.append(message)

df = pd.read_csv(RAW)
note("Loaded raw data: " + str(df.shape[0]) + " rows x " + str(df.shape[1]) + " columns")

# short nicknames for the columns I refer to a lot (the real titles are huge)
COL_SELF = "Are you self-employed?"
COL_AGE = "What is your age?"
COL_GENDER = "What is your gender?"
COL_SIZE = "How many employees does your company or organization have?"
COL_POSITION = "Which of the following best describes your work position?"
COL_COUNTRY_LIVE = "What country do you live in?"

# -----------------------------------------------------------------------------
# Step 1a: drop the free-text columns.
# -----------------------------------------------------------------------------
# The "Why or why not?" answers are opinions in people's own words -- handling
# them properly is NLP, which is a different task, so drop them. Same for the
# "if yes, what condition" follow-ups (free text, only a few people answered).
free_text = []
for column_name in df.columns:
    if column_name.startswith("Why or why not?"):
        free_text.append(column_name)

diagnosis_text = []
for column_name in df.columns:
    if column_name.startswith(("If yes, what condition",
                               "If so, what condition",
                               "If maybe, what condition")):
        diagnosis_text.append(column_name)

columns_to_drop = free_text + diagnosis_text
df = df.drop(columns=columns_to_drop)
note("Dropped " + str(len(columns_to_drop)) + " free-text columns")

# -----------------------------------------------------------------------------
# Step 1b: drop columns that are almost entirely empty.
# -----------------------------------------------------------------------------
# Some follow-ups were shown to a small group only, so they're >75% empty.
# Filling those in would mean my made-up values outnumber the real ones, so drop.
missing_fraction = df.isna().mean()
near_empty = missing_fraction[missing_fraction > 0.75].index.tolist()
df = df.drop(columns=near_empty)
note("Dropped " + str(len(near_empty)) + " columns that were more than 75% empty")

# -----------------------------------------------------------------------------
# Step 2: fix the impossible ages.
# -----------------------------------------------------------------------------
# script 00 showed ages like 3 and 323. Treat anything outside 15-80 as a typo,
# blank it, then fill all blanks with the median so one weird number can't skew
# the scaling later.
age = df[COL_AGE].copy()
how_many_bad = (~age.between(15, 80)).sum()
age[~age.between(15, 80)] = np.nan          # mark the bad ones as missing first
age = age.fillna(age.median())              # then fill all missing with median
note("Age: fixed " + str(how_many_bad) + " impossible values, set them to median "
     + str(int(age.median())))

# -----------------------------------------------------------------------------
# Step 3: tidy up the gender column (70 spellings -> 3 groups + unknown).
# -----------------------------------------------------------------------------
# rule-based sort into male/female/other. Check "female" BEFORE "male" -- the
# string "female" contains "male", and I don't want to mislabel women as men.
def map_gender(value):
    if not isinstance(value, str):
        return "unknown"                    # blank or not text
    text = value.strip().lower()
    if "female" in text or "woman" in text or text == "f":
        return "female"
    if "male" in text or "man" in text or text == "m":
        return "male"
    return "other_or_nonbinary"             # non-binary, etc.

gender_list = []
for raw_value in df[COL_GENDER]:
    gender_list.append(map_gender(raw_value))
gender = pd.Series(gender_list)
note("Gender cleaned into groups: " + str(gender.value_counts().to_dict()))

# -----------------------------------------------------------------------------
# Step 4: separate the demographics from the questions I'll cluster on.
# -----------------------------------------------------------------------------
# This is a deliberate choice. HR cares about attitudes at work, not where
# someone lives. If I leave country in, the one-hot country columns are so many
# and so strong that the clustering would mostly split people by nationality.
# So age/gender/country/self-employed go in a SEPARATE table I only use to
# describe the clusters afterwards, and come out of the main data here.
profiling = pd.DataFrame({
    "age": age,
    "gender": gender,
    "country": df[COL_COUNTRY_LIVE],
    "self_employed": df[COL_SELF],
})
demographic_columns = [COL_AGE, COL_GENDER, COL_COUNTRY_LIVE,
                       "What country do you work in?",
                       "What US state or territory do you live in?",
                       "What US state or territory do you work in?"]
demographic_columns = [c for c in demographic_columns if c in df.columns]
df = df.drop(columns=demographic_columns)
note("Set aside " + str(len(demographic_columns))
    + " demographic columns for profiling only (not for clustering)")

# -----------------------------------------------------------------------------
# Step 5a: split the multi-select job position into yes/no columns.
# -----------------------------------------------------------------------------
# people could tick several roles, joined by "|" -- str.get_dummies(sep="|")
# is made exactly for this.
positions = df[COL_POSITION].fillna("").str.get_dummies(sep="|")
new_names = []
for name in positions.columns:
    new_names.append("position_" + name.strip().replace(" ", "_").lower())
positions.columns = new_names
df = df.drop(columns=[COL_POSITION])
note("Split work position into " + str(positions.shape[1]) + " yes/no flag columns")

# -----------------------------------------------------------------------------
# Step 5b: company size is ordered, so encode it 1..6 (0 = not applicable).
# -----------------------------------------------------------------------------
# "1-5" < "6-25" < ... so keep that order instead of treating sizes as unrelated
# categories. 0 for the self-employed, who weren't asked.
size_order = {"1-5": 1, "6-25": 2, "26-100": 3,
              "100-500": 4, "500-1000": 5, "More than 1000": 6}
company_size = df[COL_SIZE].map(size_order)
company_size = company_size.fillna(0)
df = df.drop(columns=[COL_SIZE])
note("Company size encoded as an ordered number 1-6 (0 = not applicable)")

# -----------------------------------------------------------------------------
# Step 6: remaining text answers + the "not asked" blanks.
# -----------------------------------------------------------------------------
# the blanks left here are the structural ones from script 00. Don't guess --
# label them "Not applicable" so it stays a real category.
text_columns = df.select_dtypes(include=["object", "str"]).columns.tolist()
df[text_columns] = df[text_columns].fillna("Not applicable")
note("Filled the 'not asked' blanks with 'Not applicable' in "
    + str(len(text_columns)) + " text columns")

# one-hot the text columns. shorten the (very long) question titles first or the
# new column names get unmanageable.
short_prefixes = []
for column_name in text_columns:
    short_prefixes.append(column_name[:40])
dummies = pd.get_dummies(df[text_columns], prefix=short_prefixes)

# the already-numeric 0/1 flags stay as-is. one of them ("is your employer a
# tech company") still has structural blanks, but the self_employed flag already
# captures that group, so filling with 0 adds no false info.
numeric_columns = df.drop(columns=text_columns)
how_many_numeric_blanks = int(numeric_columns.isna().sum().sum())
numeric_columns = numeric_columns.fillna(0)
note("Filled " + str(how_many_numeric_blanks)
    + " structural blanks in numeric 0/1 columns with 0")

# glue it all into one numeric table
X = pd.concat([
    numeric_columns.reset_index(drop=True),
    company_size.rename("company_size_ordinal").reset_index(drop=True),
    positions.reset_index(drop=True),
    dummies.reset_index(drop=True),
], axis=1).astype(float)
note("Combined everything into a feature table: "
    + str(X.shape[0]) + " rows x " + str(X.shape[1]) + " columns")

# drop columns that are constant -- they give the algorithm nothing
unchanging = X.columns[X.var() == 0].tolist()
if len(unchanging) > 0:
    X = X.drop(columns=unchanging)
    note("Dropped " + str(len(unchanging)) + " columns that never changed")

X.to_csv("X_features.csv", index=False)
profiling.to_csv("profiling.csv", index=False)
with open("preprocessing_report.txt", "w") as f:
    f.write("\n".join(log))
note("Saved X_features.csv, profiling.csv and preprocessing_report.txt")
print("Next: try a simple first clustering in 02_first_iteration.py")
