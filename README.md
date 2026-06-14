cat > README.md << 'EOF'
# Mental Health in Tech 2016 — Unsupervised Learning Case Study

Case study for **DLBDSMLUSL01 — Machine Learning: Unsupervised Learning and
Feature Engineering** (IU), Task 1: *Mental Health in Technology-related Jobs*.

Goal: turn the messy OSMI 2016 survey (1,433 respondents × 63 questions) into
something HR can act on — clusters of similar respondents, low-dimensional
visualisations, per-cluster profiles, and programme leverage points.

## Workflow = 4 scripts = 4 report chapters

| Step | Script | What it does | Report chapter |
|---|---|---|---|
| 1. Explore | `00_exploration.py` | Descriptive stats, missingness, dirty columns | 2 |
| 2. Preprocess | `01_preprocessing.py` | Clean, engineer & encode → numeric matrix | 4 |
| 3. Quick first try | `02_first_iteration.py` | Simple PCA + k-Means — fails informatively (rediscovers survey routing) | 5 |
| 4. Step back & improve | `03_refined_iteration.py` | Employed-only, 12 attitude questions, k-Means vs GMM, 4 selection metrics | 6–8 |

## Result

Three attitude groups among the 1,146 employed respondents:
**closed & fearful (305)** — affected but silent; **uncertain (459)** —
uninformed rather than afraid; **open & supported (382)** — the internal model
of what works.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# put the Kaggle CSV under task1_data/  (filename keeps Kaggle's "heath" typo):
#   task1_data/mental-heath-in-tech-2016_20161114.csv
python3 00_exploration.py
python3 01_preprocessing.py
python3 02_first_iteration.py
python3 03_refined_iteration.py
```

## Data

OSMI Mental Health in Tech Survey 2016 —
https://www.kaggle.com/osmi/mental-health-in-tech-2016
(not committed to this repo; download separately into `task1_data/`).
EOF
