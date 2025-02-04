import logging as log
import multiprocessing
import os
import sys
import typing

import fastdtw
import numpy as np

if True:
    import common


def _distance_with_params(params: typing.Tuple[typing.Any]) -> typing.List[float]:
    worker_id, jobs = params
    # Important: fastdtw might try to use the first thread of each CPU socket, so we need to set affinity to avoid this
    os.sched_setaffinity(0, [worker_id])

    res = []
    for job in jobs:
        x, y, radius = job
        result = fastdtw.fastdtw(x, y, radius=radius)[0]
        # check if result is number and between 0 and 1
        if not isinstance(result, (int, float)):
            result = 1  # maximum distance
        elif not (0 <= result and result <= 1):
            result = 1 / (1 + np.exp(-result))  # duct taping
        res.append(result)

    return res


def _divide_length(length: int, n_worker: int) -> typing.List[int]:
    l_chunk_base = int(length / n_worker)
    l_chunk_remainder = length % n_worker
    l_chunk_each = [l_chunk_base for _ in range(n_worker)]
    for i in range(n_worker):
        if i < l_chunk_remainder:
            l_chunk_each[i] += 1

    return l_chunk_each


def _gen_DTW_distance_matrix_parallel(X: np.ndarray, radius: int = 6) -> np.ndarray:

    # X.shape = (V, T_period)

    n_V = X.shape[0]  # number of sensors

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
        _job = [(X[r], X[c], radius) for c, r in zip(*_idxs)]
        jobs[worker_id] = (worker_id, _job)
        curr_point = next_point

    with multiprocessing.Pool() as p:
        res = p.map(_distance_with_params, jobs)

    d = np.full((n_V, n_V), 1, dtype=np.float32)  # fill with maximum distance

    # flatten res keeping its order
    res = [v for chunk in res for v in chunk]

    # change to [(row, row, ...), (column, column, ...)] for numy indexing
    d[all_idxs] = res

    d = d + d.T
    np.fill_diagonal(d, 0)  # diagonal is minimum distance, so that 0
    return d


def _get_dtw_adj(data: np.ndarray, period: int) -> np.ndarray:
    len_T = data.shape[0]
    data_mean: np.ndarray = np.mean(
        [data[period*i: period*(i+1), :, 0] for i in range(len_T//period)], axis=0)
    data_mean = data_mean.squeeze().T

    '''
    dtw_distance = dtaidistance.dtw.distance_matrix(data_mean, parallel=True, compact=False, only_triu=True)
    idx_lower = np.tril_indices(dtw_distance.shape[0], -1)
    dtw_distance[idx_lower] = dtw_distance.T[idx_lower]
    np.fill_diagonal(dtw_distance, 0)
    '''

    dtw_distance = _gen_DTW_distance_matrix_parallel(data_mean)

    '''
    dtw_distance = np.zeros((self._n_vertex, self._n_vertex))

    for i in range(self._n_vertex):
        for j in range(i, self._n_vertex):
            dtw_distance[i][j] = fastdtw.fastdtw(data_mean[i], data_mean[j], radius=6)[0]

    for i in range(self._n_vertex):
        for j in range(i):
            dtw_distance[i][j] = dtw_distance[j][i]
    '''

    return dtw_distance


def main():
    assert len(
        sys.argv) == 2, 'Usage: python gen-additional-mat-STGODE.py <dataset>'

    dataset_name = sys.argv[1]
    folder_path = f'./dataset/{dataset_name}'

    _dataset = common.format.TSF(folder_path)
    log.info(f"{dataset_name=}")

    _period_unit_candidate = ['day', 'week'] + ['hour',
                                                'minute', 'second']  # this order is important!

    for _period_unit in _period_unit_candidate:
        period = _dataset.get_max_num_of_timepoints(_period_unit)
        if period > 1:
            break

    if period == 1:
        raise ValueError(
            'The period is too short to calculate the DTW distance matrix.')

    log.info("Generating dtw matrix")
    dtw_adj = _get_dtw_adj(_dataset.X, period)  # very slow!
    log.info("Completed generating similarity matrix")

    np.save(f'{folder_path}/STGODE-dtw_adj.npy', dtw_adj)
    log.info(f"Saved {folder_path}/STGODE-dtw_adj.npy")


if __name__ == '__main__':
    log.basicConfig(
        format="[%(levelname)s][%(asctime)s.%(msecs)03d][%(filename)s:%(lineno)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=log.INFO)

    main()
