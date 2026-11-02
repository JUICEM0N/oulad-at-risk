import os

import pandas as pd

from config import AT_RISK_RESULTS, CUTOFF_FRACTION, FEATURES_FILE, INTERACTION_TYPES, PROCESSED_DIR
from load_data import KEYS, load_tables

PRESENTATION = ["code_module", "code_presentation"]
NOT_FEATURES = KEYS + ["cutoff", "at_risk"]

# clicks in this many days before the cut-off count as recent activity
RECENT_DAYS = 14


def cutoff_days(courses):
    # the cut-off day of each module presentation, counted from the course start (day 0)
    cutoffs = courses[PRESENTATION].copy()
    cutoffs["cutoff"] = (courses["module_presentation_length"] * CUTOFF_FRACTION).round().astype(int)
    return cutoffs


def students(tables, cutoffs):
    info = tables["student_info"].merge(tables["student_registration"], on=KEYS).merge(cutoffs, on=PRESENTATION)
    # students who unregistered on or before the cut-off have already withdrawn, so there is nothing to predict
    still_registered = info["date_unregistration"].isna() | (info["date_unregistration"] > info["cutoff"])
    info = info[still_registered].copy()
    info["at_risk"] = info["final_result"].isin(AT_RISK_RESULTS).astype(int)
    return info[KEYS + ["cutoff", "at_risk"]]


def vle_features(tables, cutoffs):
    clicks = tables["student_vle"].merge(tables["vle"][["id_site", "activity_type"]], on="id_site")
    clicks = clicks.merge(cutoffs, on=PRESENTATION)
    clicks = clicks[clicks["date"] <= clicks["cutoff"]]
    clicks["interaction"] = clicks["activity_type"].isin(INTERACTION_TYPES)

    # engagement: course material (everything that is not a forum)
    material = clicks[~clicks["interaction"]].groupby(KEYS)
    engagement = pd.DataFrame({
        "eng_clicks": material["sum_click"].sum(),
        "eng_active_days": material["date"].nunique(),
        "eng_materials": material["id_site"].nunique(),
    })

    # interaction: discussion forums
    forum = clicks[clicks["interaction"]].groupby(KEYS)
    interaction = pd.DataFrame({
        "int_clicks": forum["sum_click"].sum(),
        "int_active_days": forum["date"].nunique(),
    })

    recent = clicks[clicks["date"] > clicks["cutoff"] - RECENT_DAYS].groupby(KEYS)["sum_click"].sum()
    before_start = clicks[clicks["date"] < 0].groupby(KEYS)["sum_click"].sum()
    other = pd.DataFrame({"recent_clicks": recent, "pre_course_clicks": before_start})

    return engagement.join(interaction, how="outer").join(other, how="outer").reset_index()


def assessment_features(tables, cutoffs):
    # assessments due on or before the cut-off (exams are at the end, so they are never included)
    due = tables["assessments"].merge(cutoffs, on=PRESENTATION)
    due = due[(due["assessment_type"] != "Exam") & (due["date"] <= due["cutoff"])]
    n_due = due.groupby(PRESENTATION).size().rename("n_due").reset_index()

    submitted = tables["student_assessment"].merge(due, on="id_assessment")
    submitted = submitted[submitted["date_submitted"] <= submitted["cutoff"]]
    submitted["late"] = submitted["date_submitted"] > submitted["date"]

    by_student = submitted.groupby(KEYS)
    features = pd.DataFrame({
        "assess_submitted": by_student.size(),
        "assess_late": by_student["late"].sum(),
        "assess_mean_score": by_student["score"].mean(),
    }).reset_index()
    return features, n_due


def build_features():
    tables = load_tables()
    cutoffs = cutoff_days(tables["courses"])
    data = students(tables, cutoffs)
    assessments, n_due = assessment_features(tables, cutoffs)

    data = data.merge(vle_features(tables, cutoffs), on=KEYS, how="left")
    data = data.merge(assessments, on=KEYS, how="left").merge(n_due, on=PRESENTATION, how="left")

    # no rows means no clicks or no submissions before the cut-off, so those counts are 0
    # assess_mean_score is also set to 0 when nothing was submitted (a tradeoff to document)
    data = data.fillna(0)
    data["assess_missed"] = data["n_due"] - data["assess_submitted"]
    data = data.drop(columns="n_due")

    # keep the label as the last column
    return data[[c for c in data.columns if c != "at_risk"] + ["at_risk"]]


def feature_columns(data):
    return [c for c in data.columns if c not in NOT_FEATURES]


if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    data = build_features()
    data.to_csv(FEATURES_FILE, index=False)
    print("Students:", len(data))
    print("At-risk:", data["at_risk"].sum(), "Not at-risk:", (data["at_risk"] == 0).sum())
    print("Features:", feature_columns(data))
    print("Saved to", FEATURES_FILE)
