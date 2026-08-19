import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from groundtruth.connectors import load_path
from groundtruth.semantic import profile
from groundtruth.store import Store


@pytest.fixture
def sample_csv() -> Path:
    return ROOT / "sample_data.csv"


@pytest.fixture
def frame(sample_csv) -> pd.DataFrame:
    return pd.read_csv(sample_csv)


@pytest.fixture
def store(sample_csv):
    s = Store()
    load_path(s, str(sample_csv), "sample")
    yield s
    s.close()


@pytest.fixture
def spec(store):
    return profile(store, "sample")
