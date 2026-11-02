from features import cutoff_days, students
from load_data import load_tables


def main():
    tables = load_tables()
    info = tables["student_info"]

    print("Courses (module presentations):", len(tables["courses"]))
    print(tables["courses"].to_string(index=False))

    print("\nStudent-course registrations:", len(info))
    print(info["final_result"].value_counts().to_string())

    print("\nFinal result by module (row %):")
    print((info.groupby("code_module")["final_result"].value_counts(normalize=True).unstack() * 100).round(1).to_string())

    print("\nAssessment types:")
    print(tables["assessments"]["assessment_type"].value_counts().to_string())

    print("\nVLE activity types (clicks):")
    clicks = tables["student_vle"].merge(tables["vle"][["id_site", "activity_type"]], on="id_site")
    print(clicks.groupby("activity_type")["sum_click"].sum().sort_values(ascending=False).to_string())

    cutoffs = cutoff_days(tables["courses"])
    kept = students(tables, cutoffs)
    print("\nStudents still registered at the cut-off:", len(kept), "of", len(info))
    print(kept["at_risk"].value_counts().rename({0: "not at-risk", 1: "at-risk"}).to_string())


if __name__ == "__main__":
    main()
