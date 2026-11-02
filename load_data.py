import os

import pandas as pd

from config import RAW_DIR

# the UCI copy marks missing values with "?", the original OU copy leaves them empty
MISSING = ["?", ""]

KEYS = ["code_module", "code_presentation", "id_student"]


def read(name, **kwargs):
    return pd.read_csv(os.path.join(RAW_DIR, name + ".csv"), na_values=MISSING, **kwargs)


def load_tables():
    tables = {
        "courses": read("courses"),
        "assessments": read("assessments"),
        "vle": read("vle"),
        "student_info": read("studentInfo"),
        "student_registration": read("studentRegistration"),
        "student_assessment": read("studentAssessment"),
    }
    # studentVle has about 10 million rows, so only the needed columns are read with small types
    tables["student_vle"] = read(
        "studentVle",
        usecols=KEYS + ["id_site", "date", "sum_click"],
        dtype={"code_module": "category", "code_presentation": "category",
               "id_student": "int32", "id_site": "int32", "date": "int16", "sum_click": "int32"},
    )
    return tables


if __name__ == "__main__":
    for name, table in load_tables().items():
        print(name, table.shape)
