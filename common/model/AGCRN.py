import math
import typing

import torch
import torch.nn.functional as F

from .__base import ModelBase


class AVWGCN(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int, cheb_k: int, embed_dim: int, device: torch.device):
        super().__init__()
        self.cheb_k = cheb_k
        self.weights_pool = torch.nn.Parameter(torch.Tensor(
            embed_dim, cheb_k, dim_in, dim_out).to(device))
        self.bias_pool = torch.nn.Parameter(
            torch.Tensor(embed_dim, dim_out).to(device))

        torch.nn.init.kaiming_uniform_(self.weights_pool, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.bias_pool, a=math.sqrt(5))

    def forward(self, x: torch.Tensor, node_embeddings: torch.Tensor) -> torch.Tensor:
        # x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        # output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        supports = F.softmax(
            F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num, device=supports.device), supports]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(
                2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        # N, cheb_k, dim_in, dim_out
        weights = torch.einsum(
            'nd,dkio->nkio', node_embeddings, self.weights_pool)
        bias = torch.matmul(node_embeddings, self.bias_pool)  # N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports,
                           x)  # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g,
                               weights) + bias  # b, N, dim_out
        return x_gconv


class AGCRNCell(torch.nn.Module):
    def __init__(self, node_num: int, dim_in: int, dim_out: int, cheb_k: int, embed_dim: int, device: torch.device):
        super().__init__()
        self.device = device
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in+self.hidden_dim, 2*dim_out,
                           cheb_k, embed_dim, device=self.device)
        self.update = AVWGCN(dim_in+self.hidden_dim, dim_out,
                             cheb_k, embed_dim, device=self.device)

    def forward(self, x: torch.Tensor, state: torch.Tensor, node_embeddings: torch.Tensor) -> torch.Tensor:
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size, self.node_num, self.hidden_dim, device=self.device)


class AVWDCRNN(torch.nn.Module):
    def __init__(self, node_num: int, dim_in: int, dim_out: int, cheb_k: int, embed_dim: int, num_layers: int, device: torch.device):
        super().__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.device = device
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = torch.nn.ModuleList()
        self.dcrnn_cells.append(
            AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim, device=self.device))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(
                AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim, device=self.device))

    def forward(self, x: torch.Tensor, init_state: torch.Tensor, node_embeddings: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](
                    current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        init_states = []
        for i in range(self.num_layers):
            init_states.append(
                self.dcrnn_cells[i].init_hidden_state(batch_size))
        # (num_layers, B, N, hidden_dim)
        return torch.stack(init_states, dim=0)


class Model(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.node_embeddings = torch.nn.Parameter(torch.randn(
            self._n_vertex, self.embed_dim, device=self._device), requires_grad=True)

        self.encoder = AVWDCRNN(self._n_vertex, self.x_dim, self.hidden_dim,
                                self.cheb_k, self.embed_dim, self.num_layers, device=self._device)

        # predictor
        self.end_conv = torch.nn.Conv2d(
            1, self.y_len * self.y_dim, kernel_size=(1, self.hidden_dim), bias=True, device=self._device)

    def forward(self, x: torch.Tensor, *_) -> torch.Tensor:
        # source: B, T_1, N, D
        # target: B, T_2, N, D
        # supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)

        # check has nan
        init_state = self.encoder.init_hidden(x.shape[0])
        output, _ = self.encoder(
            x, init_state, self.node_embeddings)  # B, T, N, hidden
        output = output[:, -1:, :, :]  # B, 1, N, hidden

        # CNN based predictor
        output: torch.Tensor = self.end_conv((output))  # B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.y_len,
                                            self.y_dim, self._n_vertex)
        output = output.permute(0, 1, 3, 2)  # B, T, N, C

        return output
