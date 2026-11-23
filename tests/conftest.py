from datetime import date, datetime

import pytest

from f1cc.analytics.lap_model import LapDeltaModel
from f1cc.etl.pipeline import run_pipeline


@pytest.fixture(scope="session")
def race_start():
    d = date.today()
    return datetime(d.year, d.month, d.day, 15)


@pytest.fixture(scope="session")
def result(race_start):
    return run_pipeline("silverstone", race_start, live_weather=False, seed=7, persist=False)


@pytest.fixture(scope="session")
def model(result):
    return LapDeltaModel(result.circuit).fit(result.laps)
