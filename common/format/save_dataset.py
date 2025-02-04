from __future__ import annotations

import os
import sqlite3
import typing

import numpy as np

# import scipy.sparse as sp

if typing.TYPE_CHECKING:
    from ._class import TSF

__all__ = [
    'save_dataset'
]


def save_dataset(tsf: TSF, dir: str):
    dir = os.path.abspath(dir)
    os.makedirs(dir, exist_ok=True)

    tsf.X = tsf.X.astype(np.float32)
    np.save(os.path.join(dir, tsf._FNAME_X), tsf.X)

    # save tensor as dense tensor
    tsf.A = tsf.A.astype(np.float32)
    np.save(os.path.join(dir, tsf._FNAME_A), tsf.A)

    # save metadata (tables)
    conn = sqlite3.connect(os.path.join(dir, tsf._FNAME_TABLE))

    # tsf.T['time'] = pd.to_datetime(tsf.T['time'])
    tsf.T.to_sql(tsf._TNAME_T, con=conn,
                 if_exists='replace', index_label='id')
    tsf.V['name'] = tsf.V['name'].astype(str)
    tsf.V.to_sql(tsf._TNAME_V, con=conn,
                 if_exists='replace', index_label='id')
    tsf.P.to_sql(tsf._TNAME_P, con=conn,
                 if_exists='replace', index_label='id')
    tsf.Q.to_sql(tsf._TNAME_Q, con=conn,
                 if_exists='replace', index_label='id')
