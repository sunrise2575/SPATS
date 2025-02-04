from __future__ import division

import sys
import typing
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

if True:
    from .__base import ModelBase


class gconv_RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        x = torch.einsum('nvc,nvw->nwc', (x, A))
        return x.contiguous()


class gconv_hyper(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        x = torch.einsum('nvc,vw->nwc', (x, A))
        return x.contiguous()


class gcn(torch.nn.Module):
    def __init__(self, dims: typing.List[int], gdep: int, alpha: float, beta: float, gamma: float, type: str = None):
        super().__init__()
        if type == 'RNN':
            self.gconv = gconv_RNN()
            self.gconv_preA = gconv_hyper()
            self.mlp = torch.nn.Linear((gdep + 1) * dims[0], dims[1])

        elif type == 'hyper':
            self.gconv = gconv_hyper()
            self.mlp = torch.nn.Sequential(
                OrderedDict([('fc1', torch.nn.Linear((gdep + 1) * dims[0], dims[1])),
                             ('sigmoid1', torch.nn.Sigmoid()),
                             ('fc2', torch.nn.Linear(dims[1], dims[2])),
                             ('sigmoid2', torch.nn.Sigmoid()),
                             ('fc3', torch.nn.Linear(dims[2], dims[3]))]))

        self.gdep = gdep
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.type_GNN = type

    def forward(self, x: torch.Tensor, adj: typing.List[torch.Tensor]) -> torch.Tensor:
        h = x
        out = [h]
        if self.type_GNN == 'RNN':
            for _ in range(self.gdep):
                h = self.alpha * x\
                    + self.beta * self.gconv(h, adj[0])\
                    + self.gamma * self.gconv_preA(h, adj[1])
                out.append(h)
        else:
            for _ in range(self.gdep):
                h = self.alpha * x\
                    + self.gamma * self.gconv(h, adj)
                out.append(h)
        ho = torch.cat(out, dim=-1)

        ho = self.mlp(ho)

        return ho


class Model(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.emb1 = torch.nn.Embedding(self._n_vertex, self.node_dim)
        self.emb2 = torch.nn.Embedding(self._n_vertex, self.node_dim)
        self.lin1 = torch.nn.Linear(self.node_dim, self.node_dim)
        self.lin2 = torch.nn.Linear(self.node_dim, self.node_dim)

        self.idx = torch.arange(self._n_vertex).to(self._device)

        dims_hyper = [self.rnn_size + self.x_dim,
                      self.hyperGNN_dim, self.middle_dim, self.node_dim]

        self.GCN1_tg = gcn(dims_hyper, self.gcn_depth,
                           *self.list_weight, 'hyper')
        self.GCN2_tg = gcn(dims_hyper, self.gcn_depth,
                           *self.list_weight, 'hyper')
        self.GCN1_tg_de = gcn(dims_hyper, self.gcn_depth,
                              *self.list_weight, 'hyper')
        self.GCN2_tg_de = gcn(dims_hyper, self.gcn_depth,
                              *self.list_weight, 'hyper')
        self.GCN1_tg_1 = gcn(dims_hyper, self.gcn_depth,
                             *self.list_weight, 'hyper')
        self.GCN2_tg_1 = gcn(dims_hyper, self.gcn_depth,
                             *self.list_weight, 'hyper')
        self.GCN1_tg_de_1 = gcn(
            dims_hyper, self.gcn_depth, *self.list_weight, 'hyper')
        self.GCN2_tg_de_1 = gcn(
            dims_hyper, self.gcn_depth, *self.list_weight, 'hyper')

        self.fc_final = torch.nn.Linear(self.rnn_size, self.y_dim)

        dims = [self.x_dim + self.rnn_size, self.rnn_size]

        self.gz1 = gcn(dims, self.gcn_depth, *self.list_weight, 'RNN')
        self.gz2 = gcn(dims, self.gcn_depth, *self.list_weight, 'RNN')
        self.gr1 = gcn(dims, self.gcn_depth, *self.list_weight, 'RNN')
        self.gr2 = gcn(dims, self.gcn_depth, *self.list_weight, 'RNN')
        self.gc1 = gcn(dims, self.gcn_depth, *self.list_weight, 'RNN')
        self.gc2 = gcn(dims, self.gcn_depth, *self.list_weight, 'RNN')

        self.gz1_de = gcn(dims, self.gcn_depth, *self.list_weight, 'RNN')
        self.gz2_de = gcn(dims, self.gcn_depth, *self.list_weight, 'RNN')
        self.gr1_de = gcn(dims, self.gcn_depth, *self.list_weight, 'RNN')
        self.gr2_de = gcn(dims, self.gcn_depth, *self.list_weight, 'RNN')
        self.gc1_de = gcn(dims, self.gcn_depth, *self.list_weight, 'RNN')
        self.gc2_de = gcn(dims, self.gcn_depth, *self.list_weight, 'RNN')

        self.use_curriculum_learning = True

    def forward(self, x: torch.Tensor, additional_info: typing.Dict[str, typing.Any]) -> torch.Tensor:
        y: torch.Tensor = additional_info['y']\
            if 'y' in additional_info else None
        batches_seen: int = additional_info['batches_seen']\
            if 'batches_seen' in additional_info else None

        batch_size = x.size(0)
        Hidden_State, Cell_State = self._initHidden(
            batch_size * self._n_vertex, self.rnn_size)

        # (B, x_len, V, x_dim) -> (B, x_dim, V, x_len)
        x = x.permute(0, 3, 2, 1)
        y = y.permute(0, 3, 2, 1)

        outputs = None
        for i in range(self.x_len):
            # Hidden_State, Cell_State = self._step(torch.squeeze(x[..., i]), Hidden_State, Cell_State, type='encoder')
            # Hidden_State, Cell_State = self._step(x[:, i, :, 0].unsqueeze(1), Hidden_State, Cell_State, type='encoder')
            Hidden_State, Cell_State = self._step(
                x[..., i], Hidden_State, Cell_State, type='encoder')

            if outputs is None:
                outputs = Hidden_State.unsqueeze(1)
            else:
                outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), 1)

        go_symbol: torch.Tensor = torch.zeros(
            (batch_size, self.y_dim, self._n_vertex), device=self._device)
        timeofday = y[:, 1:, :, :]

        decoder_input = go_symbol

        outputs_final = []

        for i in range(self.y_len):
            try:
                decoder_input = torch.cat(
                    [decoder_input, timeofday[..., i]], dim=1)
            except:
                raise ValueError('decoder_input shape: {}, timeofday shape: {}'.format(
                    decoder_input.shape, timeofday.shape))
            Hidden_State, Cell_State = self._step(
                decoder_input, Hidden_State, Cell_State, type='decoder')

            decoder_output = self.fc_final(Hidden_State)

            decoder_input = decoder_output.view(
                batch_size, self._n_vertex, self.y_dim).transpose(1, 2)
            outputs_final.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = y[:, :1, :, i]

        outputs_final = torch.stack(outputs_final, dim=1)

        outputs_final = outputs_final.view(
            batch_size, self._n_vertex, self.y_len, self.y_dim).permute(0, 2, 1, 3)

        return outputs_final

    def _preprocessing(self, adj: torch.Tensor, predefined_A: torch.Tensor) -> typing.List[torch.Tensor]:
        adj = adj + torch.eye(self._n_vertex).to(self._device)
        adj = adj / torch.unsqueeze(adj.sum(-1), -1)
        return [adj, predefined_A]

    def _step(self, input: torch.Tensor, Hidden_State: torch.Tensor, Cell_State: torch.Tensor, type: str = 'encoder') -> typing.Tuple[torch.Tensor, torch.Tensor]:
        x = input

        x = x.transpose(1, 2).contiguous()

        nodevec1 = self.emb1(self.idx)
        nodevec2 = self.emb2(self.idx)

        hyper_input = torch.cat(
            (x, Hidden_State.view(-1, self._n_vertex, self.rnn_size)), 2)

        if type == 'encoder':
            filter1 = self.GCN1_tg(hyper_input, self._supports[0])\
                + self.GCN1_tg_1(hyper_input, self._supports[1])
            filter2 = self.GCN2_tg(hyper_input, self._supports[0])\
                + self.GCN2_tg_1(hyper_input, self._supports[1])

        if type == 'decoder':
            filter1 = self.GCN1_tg_de(hyper_input, self._supports[0])\
                + self.GCN1_tg_de_1(hyper_input, self._supports[1])
            filter2 = self.GCN2_tg_de(hyper_input, self._supports[0])\
                + self.GCN2_tg_de_1(hyper_input, self._supports[1])

        nodevec1 = torch.tanh(self.tanhalpha * torch.mul(nodevec1, filter1))
        nodevec2 = torch.tanh(self.tanhalpha * torch.mul(nodevec2, filter2))

        a = torch.matmul(nodevec1, nodevec2.transpose(2, 1))\
            - torch.matmul(nodevec2, nodevec1.transpose(2, 1))

        adj = F.relu(torch.tanh(self.tanhalpha * a))

        adp = self._preprocessing(adj, self._supports[0])
        adpT = self._preprocessing(adj.transpose(1, 2), self._supports[1])

        Hidden_State = Hidden_State.view(-1, self._n_vertex, self.rnn_size)
        Cell_State = Cell_State.view(-1, self._n_vertex, self.rnn_size)

        combined = torch.cat((x, Hidden_State), -1)

        if type == 'encoder':
            z = F.sigmoid(self.gz1(combined, adp) + self.gz2(combined, adpT))
            r = F.sigmoid(self.gr1(combined, adp) + self.gr2(combined, adpT))

            temp = torch.cat((x, torch.mul(r, Hidden_State)), -1)
            Cell_State = F.tanh(self.gc1(temp, adp) + self.gc2(temp, adpT))

        elif type == 'decoder':
            z = F.sigmoid(self.gz1_de(combined, adp) +
                          self.gz2_de(combined, adpT))
            r = F.sigmoid(self.gr1_de(combined, adp) +
                          self.gr2_de(combined, adpT))

            temp = torch.cat((x, torch.mul(r, Hidden_State)), -1)
            Cell_State = F.tanh(self.gc1_de(temp, adp) +
                                self.gc2_de(temp, adpT))

        Hidden_State = torch.mul(z, Hidden_State)\
            + torch.mul(1 - z, Cell_State)

        return Hidden_State.view(-1, self.rnn_size), Cell_State.view(-1, self.rnn_size)

    def _initHidden(self, batch_size: int, hidden_size: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(
                torch.zeros(batch_size, hidden_size).to(self._device))
            Cell_State = Variable(
                torch.zeros(batch_size, hidden_size).to(self._device))

            torch.nn.init.orthogonal(Hidden_State)
            torch.nn.init.orthogonal(Cell_State)

            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, hidden_size))
            return Hidden_State, Cell_State

    def _compute_sampling_threshold(self, batches_seen: int) -> float:
        return self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))
