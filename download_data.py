import os
import urllib.request
import zipfile

from config import RAW_DIR

# Open University Learning Analytics Dataset (OULAD), UCI Machine Learning Repository mirror
# Kuzilek, J., Hlosta, M., and Zdrahal, Z. "Open University Learning Analytics dataset",
# Scientific Data 4, 170171 (2017). https://doi.org/10.1038/sdata.2017.171
URL = "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip"

FILES = ["courses.csv", "assessments.csv", "vle.csv", "studentInfo.csv",
         "studentRegistration.csv", "studentAssessment.csv", "studentVle.csv"]


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    if all(os.path.exists(os.path.join(RAW_DIR, f)) for f in FILES):
        print("OULAD is already in", RAW_DIR)
        return
    zip_path = os.path.join(RAW_DIR, "oulad.zip")
    print("Downloading OULAD (about 45 MB)...")
    urllib.request.urlretrieve(URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW_DIR)
    os.remove(zip_path)
    print("Extracted to", RAW_DIR)


if __name__ == "__main__":
    main()
