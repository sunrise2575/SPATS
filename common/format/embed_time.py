from __future__ import annotations

import typing

import numpy as np
import pandas as pd
from pandas.core.indexes.accessors import CombinedDatetimelikeProperties

if typing.TYPE_CHECKING:
    from ._class import TSF

__all__ = [
    'embed_time'
]


def year_of_total(df: CombinedDatetimelikeProperties) -> np.ndarray:
    return np.array((df.year - df.year.min()) / (df.year.max() - df.year.min() + 1))


def month_of_year(df: CombinedDatetimelikeProperties) -> np.ndarray:
    return np.array(df.month / 12)


def week_of_year(df: CombinedDatetimelikeProperties) -> np.ndarray:
    return (pd.Index(df.isocalendar()['week']) / 53).to_numpy()


def day_of_year(df: CombinedDatetimelikeProperties) -> np.ndarray:
    return np.array(df.dayofyear / 366)


def day_of_month(df: CombinedDatetimelikeProperties) -> np.ndarray:
    return np.array(df.day / df.daysinmonth)


def day_of_week(df: CombinedDatetimelikeProperties) -> np.ndarray:
    return np.array(df.dayofweek / 7)


def hour_of_day(df: CombinedDatetimelikeProperties) -> np.ndarray:
    return np.array(df.hour / 24)


def minute_of_day(df: CombinedDatetimelikeProperties) -> np.ndarray:
    return np.array((df.hour * 60 + df.minute) / (24 * 60))


def minute_of_hour(df: CombinedDatetimelikeProperties) -> np.ndarray:
    return df.minute / 60


def timestamp_in_day(df: CombinedDatetimelikeProperties) -> np.ndarray:
    T = df.time.unique()
    for i in range(len(T)):
        T[i] = T[i].strftime('T%H:%M:%S.%f')

    T = sorted(set(T))

    II = {T[i]: i for i in range(len(T))}

    E = np.ndarray((len(df.time)))
    for i, ts in enumerate(df.time):
        ts = ts.strftime('T%H:%M:%S.%f')
        E[i] = II[ts]
    return E


def timestamp_in_minute(df: CombinedDatetimelikeProperties) -> np.ndarray:
    T = df.time.unique()
    for i in range(len(T)):
        T[i] = T[i].strftime(':%M:%S.%f')

    T = sorted(set(T))

    II = {T[i]: i for i in range(len(T))}

    E = np.ndarray((len(df.time)))
    for i, ts in enumerate(df.time):
        ts = ts.strftime(':%M:%S.%f')
        E[i] = II[ts]

    return E


TEMPORAL_EMBED_FUNC: \
    typing.Dict[str, typing.Callable[[CombinedDatetimelikeProperties], np.ndarray]] \
    = {
        'year_of_total': year_of_total,
        'month_of_year': month_of_year,
        'week_of_year': week_of_year,
        'day_of_year': day_of_year,
        'day_of_month': day_of_month,
        'day_of_week': day_of_week,
        'hour_of_day': hour_of_day,
        'minute_of_day': minute_of_day,
        'minute_of_hour': minute_of_hour,
        'timestamp_in_day': timestamp_in_day,
        'timestamp_in_minute': timestamp_in_minute,
    }


def embed_time(tsf: TSF, query: str = None):
    # Temporal embed function

    if query is None:
        return

    assert query in TEMPORAL_EMBED_FUNC.keys(
    ), f'Error: query {query} is not supported'

    # stacking selected normalized temporal info
    df = tsf.T['time'].dt

    tsf.T_embedding = TEMPORAL_EMBED_FUNC.get(query)(df)
