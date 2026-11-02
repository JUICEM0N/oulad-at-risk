from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

MODEL_NAMES = ["Random Forest", "XGBoost", "Logistic Regression", "SVM", "Naive Bayes"]


def build_model(name, seed):
    # default settings for now; n_jobs=1 because the folds already run in parallel
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=seed, n_jobs=1)
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000)
    if name == "SVM":
        return SVC(kernel="rbf")
    if name == "Naive Bayes":
        return GaussianNB()
    raise ValueError("Unknown model: " + name)


def build_pipeline(name, seed):
    # SMOTE is a pipeline step, so it only runs when the pipeline is fit on the training folds.
    # Predicting on the test fold skips it, so the test fold keeps the real class balance.
    return Pipeline([
        ("scale", StandardScaler()),
        ("smote", SMOTE(random_state=seed)),
        ("model", build_model(name, seed)),
    ])
