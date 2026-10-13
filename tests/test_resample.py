"""Testy kroku 7.1: agregacja godzinowa, min_count, kompletna siatka."""
import numpy as np
import pandas as pd

from src.baseline.resample import to_hourly


def _toy() -> pd.DataFrame:
    """Jeden licznik, 3 godziny: pełna, z ujemnym odczytem, z lukąw środku."""
    ts = pd.to_datetime([
        "2020-01-01 00:15", "2020-01-01 00:30", "2020-01-01 00:45", "2020-01-01 01:00",  # h0: pełna
        "2020-01-01 01:15", "2020-01-01 01:30", "2020-01-01 01:45", "2020-01-01 02:00",  # h1: ujemny
        # h2 (02:00-03:00) brak odczytów
        "2020-01-01 03:14", "2020-01-01 03:30", "2020-01-01 03:45", "2020-01-01 04:00",  # h3: :14 spoza siatki
    ])
    cons = [1, 2, 3, 4, 1, -5, 1, 1, 2, 2, 2, 2]
    n = len(ts)
    return pd.DataFrame({
        "campus_id": 1, "meter_id": 7, "timestamp": ts, "consumption": cons,
        "campus_id_building": 1, "built_year": 2000.0, "category": "other",
        "gross_floor_area": 100.0, "room_area": np.nan, "capacity": np.nan,
        "name": "X", "capacity_campus": 10, "apparent_temperature": np.arange(n, dtype=float),
        "air_temperature": 20.0, "dew_point_temperature": 5.0, "relative_humidity": 50.0,
        "wind_speed": 1.0, "wind_direction": [350.0, 10.0] * (n // 2),
        "is_holiday": 0.0, "is_semester": 1, "is_exam": 0,
    })


def test_hourly_sum_min_count_and_grid():
    hourly, rep = to_hourly(_toy())
    assert list(hourly["timestamp"]) == list(pd.date_range("2020-01-01 00:00", periods=4, freq="h"))
    c = hourly["consumption"].tolist()
    assert c[0] == 10                      # suma 4 odczytów
    assert np.isnan(c[1])                  # ujemny -> NaN -> tylko 3 odczyty -> NaN
    assert np.isnan(c[2])                  # brak odczytów, bez forward-fill
    assert c[3] == 8                       # odczyt :14 przesunięty do :15
    assert rep["negative_set_nan"] == 1 and rep["off_grid"] == 1


def test_wind_direction_circular_mean():
    hourly, _ = to_hourly(_toy())
    wd = hourly.loc[0, "wind_direction"]
    assert min(wd, 360 - wd) < 1e-6        # średnia z 350/10 to ~0, nie 180
