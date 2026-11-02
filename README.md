# COSC 329 Project: Predicting At-Risk Students (OULAD)

Option 2, 1-person version. Predict at the midterm cut-off which students will fail or withdraw, using
the Open University Learning Analytics Dataset (OULAD), with random forest, XGBoost, logistic
regression, SVM and naive Bayes.

## Files

- config.py: paths, the cut-off fraction, which results count as at-risk, folds, trials and seed
- download_data.py: downloads OULAD into data/raw
- load_data.py: reads the 7 OULAD tables
- eda.py: prints an overview of the dataset (courses, final results, assessment and activity types)
- features.py: builds one feature vector per student from data before the cut-off, saves data/processed/features.csv
- models.py: the 5 classifiers, each in a pipeline of scaling, SMOTE, then the model
- evaluate.py: 10-fold cross-validation repeated over 25 shuffled trials, saves results/fold_scores.csv and results/summary.csv

## How to run

1. Create an environment and install the libraries:
   python -m venv .venv
   .venv\Scripts\activate          (Windows)
   pip install -r requirements.txt
2. python download_data.py
3. python eda.py
4. python features.py
5. python evaluate.py
   For a quick test: python evaluate.py --trials 1
   For some models only: python evaluate.py --models "Naive Bayes" "Logistic Regression"

evaluate.py appends each finished trial to results/fold_scores.csv and skips finished trials if it is
run again. Delete results/fold_scores.csv to start over (needed after changing the features).

## Decisions so far

- At-risk (1) is a final result of Fail or Withdrawn; Pass and Distinction are 0.
- OULAD has no midterm, so the cut-off is halfway through each module presentation (CUTOFF_FRACTION in config.py).
- Only clicks and submissions on or before the cut-off are used, so nothing from after the cut-off leaks in.
- Students who unregistered on or before the cut-off are dropped, since they have already withdrawn.
- Engagement is clicks on course material; interaction is clicks on the discussion forums (forumng).
- Students with no clicks or submissions get 0. A mean score of 0 then means nothing submitted, not a score of 0.
- SMOTE is a step inside the pipeline, so it only runs on the training folds and never on the test fold.
- Scaling is fit on the training folds only for the same reason.

## Still to decide (Week 9)

- Whether to add demographics (age band, IMD band, previous attempts, studied credits) and the module code.
- Whether to log-transform the click counts (they are heavily skewed, which hurts naive Bayes, SVM and logistic regression).
- Any other features (e.g. clicks per week, weighted assessment score).
- Model settings (all are close to the defaults right now).

## Data source

Kuzilek, J., Hlosta, M., and Zdrahal, Z. "Open University Learning Analytics dataset",
Scientific Data 4, 170171 (2017). https://doi.org/10.1038/sdata.2017.171
