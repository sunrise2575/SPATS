import logging as log
import multiprocessing
import os
import sys
import typing

import numpy as np
from scipy.optimize import linprog

if True:
    import common


def _spatial_temporal_similarity(params: typing.Tuple[typing.Any]) -> typing.List[float]:
    def _normalize(a):
        mu = np.mean(a, axis=1, keepdims=True)
        std = np.std(a, axis=1, keepdims=True)
        return (a-mu)/std

    def _wasserstein_distance(p: np.ndarray, q: np.ndarray, D: np.ndarray) -> float:
        A_eq = []
        for i in range(len(p)):
            A = np.zeros_like(D)
            A[i, :] = 1
            A_eq.append(A.reshape(-1))
        for i in range(len(q)):
            A = np.zeros_like(D)
            A[:, i] = 1
            A_eq.append(A.reshape(-1))
        A_eq = np.array(A_eq)
        b_eq = np.concatenate([p, q])
        D = np.array(D)
        D = D.reshape(-1)

        result = linprog(D, A_eq=A_eq[:-1], b_eq=b_eq[:-1])
        myresult = result.fun

        if myresult is None:
            return 0.0

        return myresult

    def _spatial_temporal_aware_distance(x: np.ndarray, y: np.ndarray) -> float:
        EPSILON = 1e-4

        x_norm = (x**2).sum(axis=1, keepdims=True)**0.5
        y_norm = (y**2).sum(axis=1, keepdims=True)**0.5

        np.place(x_norm, x_norm < EPSILON, EPSILON)
        np.place(y_norm, y_norm < EPSILON, EPSILON)

        p = x_norm[:, 0] / x_norm.sum()
        q = y_norm[:, 0] / y_norm.sum()

        D = 1 - np.dot(x / x_norm, (y / y_norm).T)

        return _wasserstein_distance(p, q, D)

    worker_id, jobs = params
    # Important: linprog tries to use the first thread of each CPU socket, so we need to set affinity to avoid this
    os.sched_setaffinity(0, [worker_id])

    res = []
    for job in jobs:
        x, y, normal, transpose = job

        if normal:
            x = _normalize(x)
            x = _normalize(x)
        if transpose:
            x = np.transpose(x)
            y = np.transpose(y)

        res.append(1 - _spatial_temporal_aware_distance(x, y))

    return res


def _divide_length(length: int, n_worker: int) -> typing.List[int]:
    l_chunk_base = int(length / n_worker)
    l_chunk_remainder = length % n_worker
    l_chunk_each = [l_chunk_base for _ in range(n_worker)]
    for i in range(n_worker):
        if i < l_chunk_remainder:
            l_chunk_each[i] += 1

    return l_chunk_each


def _gen_similarity_matrix_parallel(X: np.ndarray) -> np.ndarray:
    # X.shape = (periods, timepoints per day, V)
    n_V = X.shape[2]  # number of sensors

    d = np.zeros((n_V, n_V), dtype=np.float32)

    # create (row, column) index list
    # tuples [(row, row, ...), (column, column, ...)]
    all_idxs = np.triu_indices(n_V, k=1)

    N_IDX = len(all_idxs[0])
    N_WORKER = multiprocessing.cpu_count()
    L_CHUNK_EACH = _divide_length(N_IDX, N_WORKER)

    jobs = [None for _ in range(N_WORKER)]
    curr_point = 0
    for worker_id in range(N_WORKER):
        next_point = curr_point + L_CHUNK_EACH[worker_id]
        _idxs = [all_idxs[0][curr_point:next_point],
                 all_idxs[1][curr_point:next_point]]
        _job = [(X[:, :, r], X[:, :, c], False, False) for c, r in zip(*_idxs)]
        jobs[worker_id] = (worker_id, _job)
        curr_point = next_point

    # new version
    with multiprocessing.Pool() as p:
        res = p.map(_spatial_temporal_similarity, jobs)

    # flatten res keeping its order
    res = [v for chunk in res for v in chunk]

    # change to [(row, row, ...), (column, column, ...)] for numy indexing
    d[all_idxs] = res

    d = d + d.T
    return d


def main():
    assert len(
        sys.argv) == 3, 'Usage: python gen-additional-mat-STGODE.py <dataset> <trainTestRatio>'

    dataset_name = sys.argv[1]
    train_test_ratio = float(sys.argv[2])
    assert 0 <= train_test_ratio and train_test_ratio <= 1, 'trainTestRatio should be in [0, 1]'
    folder_path = f'./dataset/{dataset_name}'

    dataset = common.format.TSF(folder_path)
    log.info(f"{dataset_name=}, {train_test_ratio=}")

    num_samples, ndim, _ = dataset.X.shape  # num_samples = T, ndim = V
    num_train = int(num_samples * train_test_ratio)

    # decide the period
    _period_unit_candidate = ['day', 'week'] + ['hour',
                                                'minute', 'second']  # this order is important!
    for _period_unit in _period_unit_candidate:
        args_period = dataset.get_max_num_of_timepoints(_period_unit)
        # print(f"{_period_unit=}, {args_period=}")
        if args_period > 1:
            break
    if args_period == 1:
        raise ValueError(
            'The dataset is too small to generate similarity matrix')
    if args_period > num_train:
        args_period = num_train

    num_sta = int(num_train/args_period)*args_period
    data = dataset.X[:num_sta, :, :1].reshape(
        [-1, args_period, ndim])  # (periods, timepoints per day, V)

    '''
    # single core version
    d=np.zeros([ndim,ndim])
    for i in range(ndim):
        for j in range(i+1,ndim):
            #print(f"{data[:, :, i].shape=}")
            #print(f"{data[:, :, j].shape=}")
            d[i, j] = _spatial_temporal_similarity((data[:, :, i], data[:, :, j], False, False))

    sta=d+d.T
    adj = sta
    '''

    # multi core version
    log.info("Generating similarity matrix")
    adj = _gen_similarity_matrix_parallel(data)
    log.info("Completed generating similarity matrix")

    # testing version
    # adj = np.identity(ndim)

    id_mat = np.identity(ndim)
    adjl = adj + id_mat
    adjlnormd = adjl/adjl.mean(axis=0)

    adj = 1 - adjl + id_mat
    A_STAG = np.zeros([ndim, ndim])
    A_STRG = np.zeros([ndim, ndim])

    args_sparsity = 0.01
    top = int(ndim * args_sparsity)

    for i in range(adj.shape[0]):
        a = adj[i, :].argsort()[0:top]
        for j in range(top):
            A_STAG[i, a[j]] = 1
            A_STRG[i, a[j]] = adjlnormd[i, a[j]]

    for i in range(ndim):
        for j in range(ndim):
            if (i == j):
                A_STRG[i][j] = adjlnormd[i, j]

    np.save(f'{folder_path}/DSTAGNN-A_STAG-{train_test_ratio}.npy', A_STAG)
    log.info(f"Saved {folder_path}/DSTAGNN-A_STAG-{train_test_ratio}.npy")

    np.save(f'{folder_path}/DSTAGNN-A_STRG-{train_test_ratio}.npy', A_STRG)
    log.info(f"Saved {folder_path}/DSTAGNN-A_STRG-{train_test_ratio}.npy")


if __name__ == '__main__':
    log.basicConfig(
        format="[%(levelname)s][%(asctime)s.%(msecs)03d][%(filename)s:%(lineno)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=log.INFO)

    main()
