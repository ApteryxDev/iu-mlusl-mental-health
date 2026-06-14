# =============================================================================
# 01_preprocessing.py  --  Step 2 of 4: cleaning and turning text into numbers
# =============================================================================
# Course: DLBDSMLUSL01, Case Study Task 1. Goes with chapter 4 of my report
# ("Data Preparation and Feature Engineering").
#
# In the exploration (script 00) I found four problems: structural missing
# values, a messy free-text gender column, impossible ages, and free-text
# opinion columns. Here I deal with all of them and turn the survey into a
# clean table of numbers that the machine-learning libraries can actually use.
#
# My plan for this script, in order:
#   1. throw away columns I cannot use (free text, almost-empty ones)
#   2. fix the broken ages
#   3. tidy the 70 gender spellings into a few groups
#   4. put the demographics (age/gender/country) aside -- I do NOT want to
#      cluster on them, only describe clusters with them afterwards
#   5. turn the answers into numbers (ordinal where there is an order,
#      yes/no flags, and one-hot for the rest)
#   6. handle the "not asked" blanks as their own category, not a guess
#
# I keep a running list called `log` and print every decision, so that in the
# report I can show exactly what I did and how many columns each step touched.
# =============================================================================

import re
import numpy as np
import pandas as pd

RAW = "task1_data/mental-heath-in-tech-2016_20161114.csv"  # keep the "heath" typo
log = []

def note(message):
    # tiny helper: print a decision AND remember it, so I can save the whole
    # list of decisions to a text file at the end for the report.
    print(message)
    log.append(message)

df = pd.read_csv(RAW)
note("Loaded raw data: " + str(df.shape[0]) + " rows x " + str(df.shape[1]) + " columns")

# The question texts are very long, so I make short nicknames for the few
# columns I refer to a lot. This just keeps the code readable.
COL_SELF = "Are you self-employed?"
COL_AGE = "What is your age?"
COL_GENDER = "What is your gender?"
COL_SIZE = "How many employees does your company or organization have?"
COL_POSITION = "Which of the following best describes your work position?"
COL_COUNTRY_LIVE = "What country do you live in?"

# -----------------------------------------------------------------------------
# Step 1a: drop the free-text columns.
# -----------------------------------------------------------------------------
# The two "Why or why not?" columns are opinions written in people's own words.
# To analyse them properly I would need text mining (NLP), which is a different
# topic from this clustering task, so I drop them. Same for the "if yes, what
# condition" follow-ups -- they are free text and only a few people answered.
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
# Some follow-up questions were only shown to a small group, so they are more
# than 75% empty. If I tried to fill them in, my made-up values would outnumber
# the real answers, which would be dishonest. So I remove them.
missing_fraction = df.isna().mean()
near_empty = missing_fraction[missing_fraction > 0.75].index.tolist()
df = df.drop(columns=near_empty)
note("Dropped " + str(len(near_empty)) + " columns that were more than 75% empty")

# -----------------------------------------------------------------------------
# Step 2: fix the impossible ages.
# -----------------------------------------------------------------------------
# In script 00 I saw ages like 3 and 323. I decide that anything outside 15-80
# is a typing error. I replace those with the median age (a normal middle
# value), so one weird number does not distort everything later.
age = df[COL_AGE].copy()
how_many_bad = (~age.between(15, 80)).sum()
age[~age.between(15, 80)] = np.nan          # mark the bad ones as missing first
age = age.fillna(age.median())              # then fill all missing with median
note("Age: fixed " + str(how_many_bad) + " impossible values, set them to median "
     + str(int(age.median())))

# -----------------------------------------------------------------------------
# Step 3: tidy up the gender column (70 spellings -> 3 groups + unknown).
# -----------------------------------------------------------------------------
# People typed gender freely, so I write a small function with rules to sort
# each answer into male / female / other. I check "female" before "male" on
# purpose, because the word "female" contains the letters "male" and I do not
# want to mislabel women as men.
def map_gender(value):
    if not isinstance(value, str):
        return "unknown"                    # blank or not text
    text = value.strip().lower()
    if "female" in text or "woman" in text or text == "f":
        return "female"
    if "male" in text or "man" in text or text == "m":
        return "male"
    return "other_or_nonbinary"             # everything else (non-binary, etc.)

# Apply that function to every row with a plain loop, so it is obvious what
# is happening (I could use .apply(), but the loop reads more clearly).
gender_list = []
for raw_value in df[COL_GENDER]:
    gender_list.append(map_gender(raw_value))
gender = pd.Series(gender_list)
note("Gender cleaned into groups: " + str(gender.value_counts().to_dict()))

# -----------------------------------------------------------------------------
# Step 4: separate the demographics from the questions I will cluster on.
# -----------------------------------------------------------------------------
# Important choice: HR cares about attitudes at work, not about which country
# someone lives in. If I leave country in, the clustering would mostly split
# people by nationality (because one-hot country columns are many and strong).
# So I save age/gender/country/self-employed in a SEPARATE table that I only
# use later to DESCRIBE the clusters, and I remove them from the main data.
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
# Step 5a: turn the multi-select job position into simple yes/no columns.
# -----------------------------------------------------------------------------
# This question let people tick several roles, joined by "|" (e.g.
# "Back-end Developer|DevOps"). I split it into one 0/1 column per role using
# str.get_dummies, which is exactly built for this "|"-separated case.
positions = df[COL_POSITION].fillna("").str.get_dummies(sep="|")
new_names = []
for name in positions.columns:
    new_names.append("position_" + name.strip().replace(" ", "_").lower())
positions.columns = new_names
df = df.drop(columns=[COL_POSITION])
note("Split work position into " + str(positions.shape[1]) + " yes/no flag columns")

# -----------------------------------------------------------------------------
# Step 5b: company size has a natural order, so I encode it as numbers 1..6.
# -----------------------------------------------------------------------------
# "1-5" is smaller than "6-25" etc., so it makes sense to keep that order
# instead of treating the sizes as unrelated categories. I use 0 to mean
# "no company" for the self-employed (who were not asked this).
size_order = {"1-5": 1, "6-25": 2, "26-100": 3,
              "100-500": 4, "500-1000": 5, "More than 1000": 6}
company_size = df[COL_SIZE].map(size_order)
company_size = company_size.fillna(0)
df = df.drop(columns=[COL_SIZE])
note("Company size encoded as an ordered number 1-6 (0 = not applicable)")

# -----------------------------------------------------------------------------
# Step 6: deal with the remaining text answers and the "not asked" blanks.
# -----------------------------------------------------------------------------
# All the blanks still left are the structural ones from script 00 (questions
# the self-employed were never shown). Instead of guessing an answer, I label
# them literally as "Not applicable" so the model knows it was "not asked".
text_columns = df.select_dtypes(include=["object", "str"]).columns.tolist()
df[text_columns] = df[text_columns].fillna("Not applicable")
note("Filled the 'not asked' blanks with 'Not applicable' in "
    + str(len(text_columns)) + " text columns")

# Now one-hot encode the text columns: each possible answer becomes its own
# 0/1 column. The question texts are long, so I shorten them for the new
# column-name prefixes (otherwise the names get unwieldy).
short_prefixes = []
for column_name in text_columns:
    short_prefixes.append(column_name[:40])
dummies = pd.get_dummies(df[text_columns], prefix=short_prefixes)

# The columns that were already numeric (0/1 flags like self-employed) I keep
# as they are. One of them ("is your employer a tech company") still has the
# structural blanks; since the self-employed flag already captures that group,
# filling these with 0 adds no false information.
numeric_columns = df.drop(columns=text_columns)
how_many_numeric_blanks = int(numeric_columns.isna().sum().sum())
numeric_columns = numeric_columns.fillna(0)
note("Filled " + str(how_many_numeric_blanks)
    + " structural blanks in numeric 0/1 columns with 0")

# Glue everything together into one big numeric table.
X = pd.concat([
    numeric_columns.reset_index(drop=True),
    company_size.rename("company_size_ordinal").reset_index(drop=True),
    positions.reset_index(drop=True),
    dummies.reset_index(drop=True),
], axis=1).astype(float)
note("Combined everything into a feature table: "
    + str(X.shape[0]) + " rows x " + str(X.shape[1]) + " columns")

# Final tidy-up: drop any column that is the same value for everyone, because
# a column that never changes gives the algorithm no information.
unchanging = X.columns[X.var() == 0].tolist()
if len(unchanging) > 0:
    X = X.drop(columns=unchanging)
    note("Dropped " + str(len(unchanging)) + " columns that never changed")

# Save the results so the next scripts can load them.
X.to_csv("X_features.csv", index=False)
profiling.to_csv("profiling.csv", index=False)
with open("preprocessing_report.txt", "w") as f:
    f.write("\n".join(log))
note("Saved X_features.csv, profiling.csv and preprocessing_report.txt")
print("Next: try a simple first clustering in 02_first_iteration.py")
