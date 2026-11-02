import os

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
RESULTS_DIR = os.path.join(ROOT, "results")

FEATURES_FILE = os.path.join(PROCESSED_DIR, "features.csv")
FOLD_SCORES_FILE = os.path.join(RESULTS_DIR, "fold_scores.csv")
SUMMARY_FILE = os.path.join(RESULTS_DIR, "summary.csv")

# the "midterm" cut-off, as a fraction of each module presentation's length
# OULAD has no midterm, so halfway through the course is used
CUTOFF_FRACTION = 0.5

# final_result values that count as at-risk (1); Pass and Distinction are 0
AT_RISK_RESULTS = ["Fail", "Withdrawn"]

# VLE activity types that count as interaction (discussion forums); the rest are engagement
INTERACTION_TYPES = ["forumng"]

N_FOLDS = 10
N_TRIALS = 25
SEED = 329
