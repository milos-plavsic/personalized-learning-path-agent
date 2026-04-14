from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_SOURCE = (
    "UCI — Student Performance (Portuguese language course), secondary schools Portugal. "
    "https://archive.ics.uci.edu/dataset/320/student+performance"
)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_student_portuguese() -> pd.DataFrame:
    path = project_root() / "data" / "student-por.csv"
    return pd.read_csv(path, sep=";")
