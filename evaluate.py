import argparse
import os
import time

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from config import FEATURES_FILE, FOLD_SCORES_FILE, N_FOLDS, N_TRIALS, RESULTS_DIR, SEED, SUMMARY_FILE
from features import feature_columns
from models import MODEL_NAMES, build_pipeline

METRICS = ["precision", "recall", "f1", "roc_auc"]


def finished_runs():
    # the full run takes hours, so finished (model, trial) pairs are skipped if the script is restarted
    if not os.path.exists(FOLD_SCORES_FILE):
        return set()
    scores = pd.read_csv(FOLD_SCORES_FILE)
    return set(zip(scores["model"], scores["trial"]))


def run_trial(name, trial, X, y):
    # each trial reshuffles the data into new folds with a different seed
    seed = SEED + trial
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    result = cross_validate(build_pipeline(name, seed), X, y, cv=folds, scoring=METRICS, n_jobs=-1)
    rows = []
    for fold in range(N_FOLDS):
        row = {"model": name, "trial": trial, "fold": fold}
        for metric in METRICS:
            row[metric] = result["test_" + metric][fold]
        rows.append(row)
    return pd.DataFrame(rows)


def summarize():
    scores = pd.read_csv(FOLD_SCORES_FILE)
    # average over the 10 folds of each trial first, then the mean and std over the trials
    per_trial = scores.groupby(["model", "trial"])[METRICS].mean()
    summary = per_trial.groupby("model")[METRICS].agg(["mean", "std"])
    summary.columns = [metric + "_" + stat for metric, stat in summary.columns]
    summary["trials"] = per_trial.groupby("model").size()
    summary.to_csv(SUMMARY_FILE)
    print(summary.round(3).to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    args = parser.parse_args()

    data = pd.read_csv(FEATURES_FILE)
    X = data[feature_columns(data)]
    y = data["at_risk"]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    done = finished_runs()

    for name in args.models:
        for trial in range(args.trials):
            if (name, trial) in done:
                continue
            start = time.time()
            scores = run_trial(name, trial, X, y)
            scores.to_csv(FOLD_SCORES_FILE, mode="a", index=False, header=not os.path.exists(FOLD_SCORES_FILE))
            print(f"{name} trial {trial + 1}/{args.trials}: f1 {scores['f1'].mean():.3f}, {time.time() - start:.0f}s")

    summarize()


if __name__ == "__main__":
    main()
