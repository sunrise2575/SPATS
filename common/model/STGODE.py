import math
import multiprocessing
import os
import typing

import fastdtw
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Whether use adjoint method or not.
adjoint = False
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

if True:
    from .__base import ModelBase


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


# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
class ODEFunc(nn.Module):
    def __init__(self, feature_dim: int, temporal_dim: int, adj: torch.Tensor):
        super().__init__()
        self.adj = adj
        self.x0 = None
        self.alpha = nn.Parameter(0.8 * torch.ones(adj.shape[1]))
        self.beta = 0.6
        self.w = nn.Parameter(torch.eye(feature_dim))
        self.d = nn.Parameter(torch.zeros(feature_dim) + 1)
        self.w2 = nn.Parameter(torch.eye(temporal_dim))
        self.d2 = nn.Parameter(torch.zeros(temporal_dim) + 1)

    def forward(self, t, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(
            self.alpha).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        xa = torch.einsum('ij, kjlm->kilm', self.adj, x)

        # ensure the eigenvalues to be less than 1
        d = torch.clamp(self.d, min=0, max=1)
        w = torch.mm(self.w * d, torch.t(self.w))
        xw = torch.einsum('ijkl, lm->ijkm', x, w)

        d2 = torch.clamp(self.d2, min=0, max=1)
        w2 = torch.mm(self.w2 * d2, torch.t(self.w2))
        xw2 = torch.einsum('ijkl, km->ijml', x, w2)

        f = alpha / 2 * xa - x + xw - x + xw2 - x + self.x0
        return f


class ODEblock(nn.Module):
    def __init__(self, odefunc: ODEFunc, t=torch.tensor([0, 1])):
        super().__init__()
        self.t = t
        self.odefunc = odefunc

    def set_x0(self, x0: torch.Tensor) -> None:
        self.odefunc.x0 = x0.clone().detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.t.type_as(x)
        z = odeint(self.odefunc, x, t, method='euler')[1]
        return z


# Define the ODEGCN model.
class ODEG(nn.Module):
    def __init__(self, feature_dim: int, temporal_dim: int, adj: torch.Tensor, time: int):
        super().__init__()
        self.odeblock = ODEblock(
            ODEFunc(feature_dim, temporal_dim, adj), t=torch.tensor([0, time]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.odeblock.set_x0(x)
        z = self.odeblock(x)
        return F.relu(z)


class Chomp1d(nn.Module):
    """
    extra dimension will be added by padding, remove it
    """

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :, :-self.chomp_size].contiguous()


class TemporalConvNet(nn.Module):
    """
    time dilation convolution
    """

    def __init__(self, num_inputs: int, num_channels: int, kernel_size: int = 2, dropout: float = 0.2):
        """
        Args:
            num_inputs : channel's number of input data's feature
            num_channels : numbers of data feature tranform channels, the last is the output channel
            kernel_size : using 1d convolution, so the real kernel is (1, kernel_size) 
        """
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            self.conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), dilation=(
                1, dilation_size), padding=(0, padding))
            self.conv.weight.data.normal_(0, 0.01)
            self.chomp = Chomp1d(padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

            layers += [nn.Sequential(self.conv, self.chomp,
                                     self.relu, self.dropout)]

        self.network = nn.Sequential(*layers)
        self.downsample = nn.Conv2d(
            num_inputs, num_channels[-1], (1, 1)) if num_inputs != num_channels[-1] else None
        if self.downsample:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        like ResNet
        Args:
            X : input data of shape (B, N, T, F) 
        """
        # permute shape to (B, F, N, T)
        y = x.permute(0, 3, 1, 2)
        y = F.relu(self.network(y) + self.downsample(y)
                   if self.downsample else y)
        y = y.permute(0, 2, 3, 1)
        return y


class GCN(nn.Module):
    def __init__(self, A_hat: torch.Tensor, in_channels: int, out_channels: int):
        super().__init__()
        self.A_hat = A_hat
        self.theta = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.reset()

    def reset(self) -> None:
        stdv = 1. / math.sqrt(self.theta.shape[1])
        self.theta.data.uniform_(-stdv, stdv)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        y = torch.einsum('ij, kjlm-> kilm', self.A_hat, X)
        return F.relu(torch.einsum('kjlm, mn->kjln', y, self.theta))


class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, x_len: int, num_nodes: int, A_hat: torch.Tensor):
        """
        Args:
            in_channels: Number of input features at each node in each time step.
            out_channels: a list of feature channels in timeblock, the last is output feature channel
            num_nodes: Number of nodes in the graph
            A_hat: the normalized adjacency matrix
        """
        super().__init__()
        self.A_hat = A_hat
        self.temporal1 = TemporalConvNet(num_inputs=in_channels,
                                         num_channels=out_channels)
        self.odeg = ODEG(out_channels[-1], x_len, A_hat, time=6)
        self.temporal2 = TemporalConvNet(num_inputs=out_channels[-1],
                                         num_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: Input data of shape (batch_size, num_nodes, num_timesteps, num_features)
        Return:
            Output data of shape(batch_size, num_nodes, num_timesteps, out_channels[-1])
        """
        t = self.temporal1(X)
        t = self.odeg(t)
        t = self.temporal2(F.relu(t))

        return self.batch_norm(t)


class Model(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        A = self._supports[0].detach().cpu().numpy()

        self.A_sp_hat = self._get_normalized_adj(A)

        _period_unit_candidate = ['day', 'week'] + ['hour',
                                                    'minute', 'second']  # this order is important!

        for _period_unit in _period_unit_candidate:
            period = self._dataset.get_max_num_of_timepoints(_period_unit)
            if period > 1:
                break

        if period == 1:
            raise ValueError(
                'The period is too short to calculate the DTW distance matrix.')

        '''
        # for testing
        self.A_se_hat = torch.eye(self._n_vertex, device=self._device)
        '''

        dtw_path = os.path.abspath(
            f'{self._dataset.dataset_path}/STGODE-dtw_adj.npy')
        if os.path.exists(dtw_path) and os.path.isfile(dtw_path):
            dtw_adj = np.load(dtw_path)
        else:
            dtw_adj = self._get_dtw_adj(self._dataset.X, period)  # very slow!
        self.A_se_hat = self._get_normalized_adj(dtw_adj)

        # spatial graph
        self.sp_blocks = nn.ModuleList(
            [nn.Sequential(
                STGCNBlock(in_channels=self.x_dim, out_channels=[64, 32, 64],
                           x_len=self.x_len,
                           num_nodes=self._n_vertex, A_hat=self.A_sp_hat),
                STGCNBlock(in_channels=64, out_channels=[64, 32, 64],
                           x_len=self.x_len,
                           num_nodes=self._n_vertex, A_hat=self.A_sp_hat)) for _ in range(3)
             ])
        # semantic graph
        self.se_blocks = nn.ModuleList([nn.Sequential(
            STGCNBlock(in_channels=self.x_dim, out_channels=[64, 32, 64],
                       x_len=self.x_len,
                       num_nodes=self._n_vertex, A_hat=self.A_se_hat),
            STGCNBlock(in_channels=64, out_channels=[64, 32, 64],
                       x_len=self.x_len,
                       num_nodes=self._n_vertex, A_hat=self.A_se_hat)) for _ in range(3)
        ])

        self.pred = nn.Sequential(
            nn.Linear(self.x_len * 64, self.y_len * 32),
            nn.ReLU(),
            nn.Linear(self.y_len * 32, self.y_len)
        )

    def _get_normalized_adj(self, A: np.ndarray) -> torch.Tensor:
        """
        Returns a tensor, the degree normalized adjacency matrix.
        """
        alpha = 0.8
        D = np.array(np.sum(A, axis=1)).reshape((-1,))
        D[D <= 10e-5] = 10e-5    # Prevent infs
        diag = np.reciprocal(np.sqrt(D))
        A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                             diag.reshape((1, -1)))
        A_reg = alpha / 2 * (np.eye(A.shape[0]) + A_wave)
        return torch.from_numpy(A_reg.astype(np.float32)).to(self._device)

    def _get_dtw_adj(self, data: np.ndarray, period: int) -> np.ndarray:
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

    def forward(self, x: torch.Tensor, *_) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3)  # (B, T, V, x_dim) -> (B, V, T, x_dim)
        outs = []
        # spatial graph
        for blk in self.sp_blocks:
            outs.append(blk(x))
        # semantic graph
        for blk in self.se_blocks:
            outs.append(blk(x))
        outs = torch.stack(outs)
        x = torch.max(outs, dim=0)[0]
        x = x.reshape((x.shape[0], x.shape[1], -1))

        x = self.pred(x)
        # (B, V, T) -> (B, T, V) -> (B, T, V, y_dim)
        x = x.permute(0, 2, 1).unsqueeze(-1)
        return x
