import os
import typing

import numpy as np
import pandas as pd
import torch
import torch.utils.data

if True:
    from .embed_time import *
    from .load_dataset import *
    from .save_dataset import *

__all__ = [
    'TSF',
    'TSFMovingWindow'
]


class TSF(torch.utils.data.Dataset):
    # tensor name
    _FNAME_X = 'X.npy'
    _FNAME_A = 'A.npy'

    # table name
    _FNAME_TABLE = 'table.sqlite3'
    _TNAME_T = 'timestamp'
    _TNAME_V = 'vertex_info'
    _TNAME_P = 'vertex_value_info'
    _TNAME_Q = 'edge_value_info'

    def __init__(self, folder_path: str = None):
        super().__init__()

        # file name
        if folder_path is not None:
            self.load(folder_path)
            self.dataset_path = os.path.abspath(folder_path)

    def __getitem__(self, i: int) -> torch.Tensor:
        return torch.from_numpy(self.X[i, ...])

    def __len__(self) -> int:
        return self.X.shape[0]

    def info(self):
        print(f'{self.X.shape=}')
        print(f'{self.A.shape=}')
        print(f'{self.P.shape=}')
        print(f'{self.Q.shape=}')
        print(f'{self.T.shape=}')
        print(f'{self.V.shape=}')

    def reset(self):
        self.X: np.ndarray = None
        self.A: np.ndarray = None

        self.P: pd.DataFrame = None
        self.Q: pd.DataFrame = None
        self.T: pd.DataFrame = None
        self.V: pd.DataFrame = None

        self.X_mean: float = 0
        self.X_std: float = 0

        self.T_embedding: np.ndarray = None

    def load(self, dir: str):
        load_dataset(self, dir)

    def save(self, dir: str):
        save_dataset(self, dir)

    def embed_time(self, query: str):
        embed_time(self, query)

    def get_max_num_of_timepoints(self, period: str = 'day') -> int:
        _period_candidate = ['year', 'month', 'week',
                             'day', 'hour', 'minute', 'second']

        period = period.lower()
        assert period in _period_candidate, f'period should be one of {_period_candidate}'

        # _temp = pd.to_datetime(self.T['time'])
        _temp = pd.to_datetime(self.T['time'], format='mixed')
        if period == 'year':
            return _temp.groupby(_temp.dt.year).size().max()
        if period == 'month':
            return _temp.groupby(_temp.dt.month).size().max()
        if period == 'week':
            return _temp.groupby(_temp.dt.isocalendar().week).size().max()
        if period == 'day':
            return _temp.groupby(_temp.dt.date).size().max()
        if period == 'hour':
            return _temp.groupby(_temp.dt.hour).size().max()
        if period == 'minute':
            return _temp.groupby(_temp.dt.minute).size().max()
        if period == 'second':
            return _temp.groupby(_temp.dt.second).size().max()

    def leave_only_this_columns(self, column_name: typing.List[str]):
        # 선택한 컬럼들만 남기는 함수
        idxs = []
        for name in column_name:
            idx = self.P[self.P['name'] == name].index.values
            if len(idx) == 1:
                idxs.append(int(idx))
            else:
                continue

        self.X = self.X[:, :, idxs]
        self.P = self.P.iloc[idxs]


class TSFMovingWindow(TSF):
    def __init__(self, x_len: int = 12, y_len: int = 12, **kwargs):
        super().__init__(**kwargs)
        self.x_len = x_len
        self.y_len = y_len

    def __len__(self) -> int:
        return self.X.shape[0] - (self.x_len + self.y_len) + 1

    def __getitem__(self, i: int) -> typing.Dict[str, torch.Tensor]:
        x_s, x_e = i, i + self.x_len
        y_s, y_e = i + self.x_len, i + self.x_len + self.y_len

        if self.T_embedding is None:
            return {
                'x': torch.from_numpy(self.X[x_s:x_e, :, :]),
                'y': torch.from_numpy(self.X[y_s:y_e, :, :]),
            }
        else:
            return {
                'x': torch.from_numpy(self.X[x_s:x_e, :, :]),
                'y': torch.from_numpy(self.X[y_s:y_e, :, :]),
                'x_time': torch.from_numpy(self.T_embedding[x_s:x_e]),
                'y_time': torch.from_numpy(self.T_embedding[y_s:y_e]),
            }
