from __future__ import annotations

import os
import sqlite3
import typing

import numpy as np
import pandas as pd

# import scipy.sparse as sp
# import operator

if typing.TYPE_CHECKING:
    from ._class import TSF

__all__ = [
    'load_dataset'
]


def load_dataset(tsf: TSF, dir: str):
    tsf.reset()

    dir = os.path.abspath(dir)

    tsf.X = np.load(os.path.join(dir, tsf._FNAME_X))
    tsf.A = np.load(os.path.join(dir, tsf._FNAME_A))

    # load metadata (tables)
    conn = sqlite3.connect(os.path.join(dir, tsf._FNAME_TABLE))

    tsf.T = pd.read_sql(
        f'SELECT * FROM {tsf._TNAME_T}', con=conn, index_col='id')
    tsf.T.index.name = None
    tsf.T['time'] = pd.to_datetime(tsf.T['time'], format='mixed')

    tsf.V = pd.read_sql(
        f'SELECT * FROM {tsf._TNAME_V}', con=conn, index_col='id')
    tsf.V.index.name = None
    tsf.V['name'] = tsf.V['name'].astype("string")

    tsf.P = pd.read_sql(
        f'SELECT * FROM {tsf._TNAME_P}', con=conn, index_col='id')
    tsf.P.index.name = None

    tsf.Q = pd.read_sql(
        f'SELECT * FROM {tsf._TNAME_Q}', con=conn, index_col='id')
    tsf.Q.index.name = None
