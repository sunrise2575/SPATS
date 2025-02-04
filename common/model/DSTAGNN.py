# -*- coding:utf-8 -*-
import math
import multiprocessing
import os
import typing

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linprog

if True:
    from .. import format
    from .__base import ModelBase


def _spatial_temporal_similarity(params: typing.Tuple[typing.Any]) -> typing.List[float]:
    def _normalize(a):
        mu = np.mean(a, axis=1, keepdims=True)
        std = np.std(a, axis=1, keepdims=True)
        return (a - mu) / std

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


class SScaledDotProductAttention(torch.nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q: torch.Tensor, K: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(
            self.d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        if attn_mask is not None:
            # Fills elements of self tensor with value where mask is True.
            scores.masked_fill_(attn_mask, -1e9)
        return scores


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, d_k: int, num_of_d: int):
        super().__init__()
        self.d_k = d_k
        self.num_of_d = num_of_d

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, attn_mask: torch.Tensor, res_att: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k) + \
            res_att  # scores : [batch_size, n_heads, len_q, len_k]
        if attn_mask is not None:
            # Fills elements of self tensor with value where mask is True.
            scores.masked_fill_(attn_mask, -1e9)
        attn = F.softmax(scores, dim=3)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, scores


class SMultiHeadAttention(torch.nn.Module):
    def __init__(self, DEVICE: torch.device, d_model: int, d_k: int, d_v: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.DEVICE = DEVICE
        self.W_Q = torch.nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = torch.nn.Linear(d_model, d_k * n_heads, bias=False)

    def forward(self, input_Q: torch.Tensor, input_K: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        # Q: [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q.forward(input_Q).view(
            batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k]
        K = self.W_K.forward(input_K).view(
            batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1,
                                                      1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        attn = SScaledDotProductAttention(self.d_k)(Q, K, attn_mask)
        return attn


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, DEVICE: torch.device, d_model: int, d_k: int, d_v: int, n_heads: int, num_of_d: int):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.num_of_d = num_of_d
        self.DEVICE = DEVICE
        self.W_Q = torch.nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = torch.nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = torch.nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = torch.nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q: torch.Tensor, input_K: torch.Tensor, input_V: torch.Tensor, attn_mask: torch.Tensor, res_att: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q.forward(input_Q).view(batch_size, self.num_of_d, -1, self.n_heads,
                                           self.d_k).transpose(2, 3)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K.forward(input_K).view(batch_size, self.num_of_d, -1, self.n_heads,
                                           self.d_k).transpose(2, 3)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V.forward(input_V).view(batch_size, self.num_of_d, -1, self.n_heads,
                                           self.d_v).transpose(2, 3)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1,
                                                      1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, res_attn = ScaledDotProductAttention(
            self.d_k, self.num_of_d).forward(Q, K, V, attn_mask, res_att)

        context = context.transpose(2, 3).reshape(batch_size, self.num_of_d, -1,
                                                  self.n_heads * self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]

        return torch.nn.LayerNorm(self.d_model).to(self.DEVICE)(output + residual), res_attn


class cheb_conv_withSAt(torch.nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K: int, cheb_polynomials: typing.List[torch.Tensor], in_channels: int, out_channels: int, num_of_vertices: int):
        '''
        :param K: int
        :param in_channees: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super().__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.relu = torch.nn.ReLU(inplace=True)
        self.Theta = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])
        self.mask = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x: torch.Tensor, spatial_attention: torch.Tensor, adj_pa: torch.Tensor) -> torch.Tensor:
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(
                self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)
                mask = self.mask[k]

                myspatial_attention = spatial_attention[:,
                                                        k, :, :] + adj_pa.mul(mask)
                myspatial_attention = F.softmax(myspatial_attention, dim=1)

                # (N,N)*(N,N) = (N,N) 多行和为1, 按着列进行归一化
                T_k_with_at = T_k.mul(myspatial_attention)

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                # (N, N)(b, N, F_in) = (b, N, F_in) 因为是左乘，所以多行和为1变为多列和为1，即一行之和为1，进行左乘
                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)

                # (b, N, F_in)(F_in, F_out) = (b, N, F_out)
                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)

        return self.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)


class cheb_conv(torch.nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K: int, cheb_polynomials: typing.List[torch.Tensor], in_channels: int, out_channels: int):
        '''
        :param K: int
        :param in_channees: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super().__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(
            in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(
                self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = graph_signal.permute(
                    0, 2, 1).matmul(T_k).permute(0, 2, 1)

                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))


class Embedding(torch.nn.Module):
    def __init__(self, nb_seq: int, d_Em: int, num_of_features: int, Etype: str, device: torch.device):
        super().__init__()
        self._device = device
        self.nb_seq = nb_seq
        self.Etype = Etype
        self.num_of_features = num_of_features
        self.pos_embed = torch.nn.Embedding(nb_seq, d_Em)
        self.norm = torch.nn.LayerNorm(d_Em)

    def forward(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        if self.Etype == 'T':
            pos = torch.arange(self.nb_seq, dtype=torch.long).to(self._device)
            pos = pos.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_of_features,
                                                       self.nb_seq)  # [seq_len] -> [batch_size, seq_len]
            embedding = x.permute(0, 2, 3, 1) + self.pos_embed(pos)
        else:
            pos = torch.arange(self.nb_seq, dtype=torch.long).to(self._device)
            pos = pos.unsqueeze(0).expand(batch_size, self.nb_seq)
            embedding = x + self.pos_embed(pos)
        Emx = self.norm(embedding)
        return Emx


class GTU(torch.nn.Module):
    def __init__(self, in_channels: int, time_strides: int, kernel_size: int):
        super().__init__()
        self.in_channels = in_channels
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.con2out = torch.nn.Conv2d(
            in_channels, 2 * in_channels, kernel_size=(1, kernel_size), stride=(1, time_strides))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_causal_conv = self.con2out(x)
        x_p = x_causal_conv[:, : self.in_channels, :, :]
        x_q = x_causal_conv[:, -self.in_channels:, :, :]
        x_gtu = torch.mul(self.tanh(x_p), self.sigmoid(x_q))
        return x_gtu


class DSTAGNN_block(torch.nn.Module):
    def __init__(self, DEVICE: torch.device, num_of_d: int, in_channels: int, K: int, nb_chev_filter: int, nb_time_filter: int, time_strides: int,
                 cheb_polynomials: typing.List[torch.Tensor], adj_pa: torch.Tensor, adj_TMD: torch.Tensor, num_of_vertices: int, num_of_timesteps: int, d_model: int, d_k: int, d_v: int, n_heads: int):
        super().__init__()

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU(inplace=True)

        self.adj_pa = torch.FloatTensor(adj_pa).to(DEVICE)

        self.pre_conv = torch.nn.Conv2d(
            num_of_timesteps, d_model, kernel_size=(1, num_of_d))

        self.EmbedT = Embedding(
            num_of_timesteps, num_of_vertices, num_of_d, 'T', device=DEVICE)
        self.EmbedS = Embedding(num_of_vertices, d_model,
                                num_of_d, 'S', device=DEVICE)

        self.TAt = MultiHeadAttention(
            DEVICE, num_of_vertices, d_k, d_v, n_heads, num_of_d)
        self.SAt = SMultiHeadAttention(DEVICE, d_model, d_k, d_v, K)

        self.cheb_conv_SAt = cheb_conv_withSAt(
            K, cheb_polynomials, in_channels, nb_chev_filter, num_of_vertices)

        self.gtu3 = GTU(nb_time_filter, time_strides, 3)
        self.gtu5 = GTU(nb_time_filter, time_strides, 5)
        self.gtu7 = GTU(nb_time_filter, time_strides, 7)
        self.pooling = torch.torch.nn.MaxPool2d(kernel_size=(1, 2), stride=None, padding=0,
                                                return_indices=False, ceil_mode=False)

        self.residual_conv = torch.nn.Conv2d(
            in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))

        self.dropout = torch.nn.Dropout(p=0.05)
        self.fcmy = torch.nn.Sequential(
            torch.nn.Linear(3 * num_of_timesteps - 12, num_of_timesteps),
            torch.nn.Dropout(0.05),
        )
        self.ln = torch.nn.LayerNorm(nb_time_filter)

    def forward(self, x: torch.Tensor, res_att: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        '''
        :param x: (Batch_size, N, F_in, T)
        :param res_att: (Batch_size, N, F_in, T)
        :return: (Batch_size, N, nb_time_filter, T)
        '''
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape  # B,N,F,T

        # TAT
        if num_of_features == 1:
            TEmx = self.EmbedT(x, batch_size)  # B,F,T,N
        else:
            TEmx = x.permute(0, 2, 3, 1)
        TATout, re_At = self.TAt.forward(
            TEmx, TEmx, TEmx, None, res_att)  # B,F,T,N; B,F,Ht,T,T

        x_TAt = self.pre_conv(TATout.permute(0, 2, 3, 1))[
            :, :, :, -1].permute(0, 2, 1)  # B,N,d_model

        # SAt
        SEmx_TAt = self.EmbedS(x_TAt, batch_size)  # B,N,d_model
        SEmx_TAt = self.dropout(SEmx_TAt)   # B,N,d_model
        STAt = self.SAt(SEmx_TAt, SEmx_TAt, None)  # B,Hs,N,N

        # graph convolution in spatial dim
        spatial_gcn = self.cheb_conv_SAt.forward(
            x, STAt, self.adj_pa)  # B,N,F,T

        # convolution along the time axis
        X = spatial_gcn.permute(0, 2, 1, 3)  # B,F,N,T
        x_gtu = []
        x_gtu.append(self.gtu3(X))  # B,F,N,T-2
        x_gtu.append(self.gtu5(X))  # B,F,N,T-4
        x_gtu.append(self.gtu7(X))  # B,F,N,T-6
        time_conv = torch.cat(x_gtu, dim=-1)  # B,F,N,3T-12
        time_conv = self.fcmy(time_conv)

        if num_of_features == 1:
            time_conv_output = self.relu(time_conv)
        else:
            time_conv_output = self.relu(X + time_conv)  # B,F,N,T

        # residual shortcut
        if num_of_features == 1:
            x_residual = self.residual_conv(x.permute(0, 2, 1, 3))
        else:
            x_residual = x.permute(0, 2, 1, 3)
        x_residual = self.ln(
            F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)

        return x_residual, re_At


class Model(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        STAG_path = os.path.abspath(
            f"{self._dataset.dataset_path}/DSTAGNN-A_STAG-{self._ctx['trainTestRatio']}.npy")
        STRG_path = os.path.abspath(
            f"{self._dataset.dataset_path}/DSTAGNN-A_STRG-{self._ctx['trainTestRatio']}.npy")

        if (os.path.exists(STAG_path) and os.path.isfile(STAG_path))\
                and (os.path.exists(STRG_path) and os.path.isfile(STRG_path)):
            self.adj_TMD = np.load(STAG_path)
            self.adj_pa = np.load(STRG_path)
        else:
            self.adj_TMD, self.adj_pa = self._gen_STAG_and_STRG(self._dataset)

        self.K = len(self._supports)

        self.BlockList = torch.nn.ModuleList([
            DSTAGNN_block(
                self._device,
                self.x_dim, self.x_dim, self.K, self.nb_chev_filter, self.nb_time_filter, 1, self._supports, self.adj_pa, self.adj_TMD, self._n_vertex, self.x_len, self.d_model, self.d_k, self.d_k, self.n_heads)])

        self.BlockList.extend([
            DSTAGNN_block(
                self._device,
                self.x_dim * self.nb_time_filter, self.nb_chev_filter, self.K, self. nb_chev_filter, self.nb_time_filter, 1, self._supports, self.adj_pa, self.adj_TMD, self._n_vertex, self.x_len, self.d_model, self.d_k, self.d_k, self.n_heads)
            for _ in range(self.nb_block - 1)])

        self.final_conv = torch.nn.Conv2d(int(
            (self.x_len) * self.nb_block), 128, kernel_size=(1, self.nb_time_filter), device=self._device)

        self.final_fc = torch.nn.Linear(128, self.y_len, device=self._device)

        for p in self.parameters():
            p: torch.nn.Parameter
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.uniform_(p)

    def _gen_STAG_and_STRG(self, dataset: format.TSFMovingWindow) -> typing.Tuple[np.ndarray, np.ndarray]:
        num_samples, ndim, _ = dataset.X.shape  # num_samples = T, ndim = V
        num_train = int(num_samples * self._ctx['trainTestRatio'])

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

        num_sta = int(num_train / args_period) * args_period
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
        adj = _gen_similarity_matrix_parallel(data)

        # testing version
        # adj = np.identity(ndim)

        id_mat = np.identity(ndim)
        adjl = adj + id_mat
        adjlnormd = adjl / adjl.mean(axis=0)

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

        return A_STAG, A_STRG

    def forward(self, x: torch.Tensor, *_) -> torch.Tensor:
        # [B, T, V, d_X] -> [B, V, d_X, T]
        x = x.permute(0, 2, 3, 1)

        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        """
        for block in self.BlockList:
            x = block(x)
            
        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        """

        need_concat = []
        res_att = 0
        for block in self.BlockList:
            x, res_att = block(x, res_att)
            need_concat.append(x)

        final_x = torch.cat(need_concat, dim=-1)
        output1 = self.final_conv.forward(final_x.permute(0, 3, 1, 2))[
            :, :, :, -1].permute(0, 2, 1)
        output = self.final_fc(output1)

        # [B, V, T] -> [B, V, T, 1] -> [B, T, V, 1]
        output = output.unsqueeze(-1).permute(0, 2, 1, 3)

        return output
