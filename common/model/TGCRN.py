import typing

import numpy as np
import torch
import torch.nn.functional as F

from .__base import ModelBase


def dynamic_topK(A: torch.Tensor, k: int = 10) -> torch.Tensor:
    """
    A [batch, node, node]
    """
    mask = torch.zeros_like(A, device=A.device)
    mask.fill_(float('0'))
    s1, t1 = (A + torch.rand_like(A, device=A.device)*0.01).topk(k, 2)
    mask.scatter_(2, t1, s1.fill_(1))
    A = A * mask

    return A


class TGCN(torch.nn.Module):
    """
    ChebNet with node_embeddings [batch, node, dim]
    """

    def __init__(self, in_dim: int, out_dim: int, cheb_k: int, embed_dim: int, device: torch.device, period: bool = False):
        super().__init__()
        self.device = device
        self.cheb_k = cheb_k
        # learned weights
        self.weights_pool = torch.nn.Parameter(
            torch.FloatTensor(embed_dim, cheb_k, in_dim, out_dim).to(self.device))
        self.bias_pool = torch.nn.Parameter(
            torch.FloatTensor(embed_dim, out_dim).to(self.device))
        self.period = period
        self.init_weights()

    def forward(self, x: torch.Tensor, node_embeddings: torch.Tensor) -> torch.Tensor:
        # x [x;hidden_state]
        # node_embedding [n, d]  or dynamic [b, n, d]
        # add temporal variance
        bs = x.shape[0]
        node_num = x.shape[1]
        node_embeddings, t, n_t, p = node_embeddings
        t_d = t.size(-1)

        # node embedding
        a = torch.mm(node_embeddings, node_embeddings.transpose(
            0, 1)).repeat(x.size(0), 1, 1)  # b,n,n
        # time embedding
        a_t = torch.bmm(n_t.view(bs, 1, t_d), t.view(bs, t_d, 1))
        # fusion: add, concat, mlp
        _a = a + a_t

        if self.period:
            A = F.softmax(dynamic_topK(
                F.relu((1+0.3*torch.sigmoid(p))*_a), 10), dim=2)
        else:
            A = F.softmax(F.relu(_a), dim=2)

        # A = F.softmax(F.relu(_a), dim=2)
        I = torch.eye(node_num, device=self.device).repeat(x.size(0), 1, 1)
        support_set = [I, A]
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(
                2 * A, support_set[-1]) - support_set[-2])

        # node represention and time represenation
        node_embeddings = node_embeddings.repeat(x.size(0), 1, 1)
        node_embeddings = torch.cat(
            (node_embeddings, n_t.unsqueeze(dim=1).repeat(1, node_num, 1)), dim=-1)

        cheb_supports = torch.stack(support_set, dim=1)
        # N, cheb_k, dim_in, dim_out
        weights = torch.einsum(
            'bnd,dkio->bnkio', node_embeddings, self.weights_pool)
        bias = torch.matmul(node_embeddings, self.bias_pool)  # B, N, dim_out
        x_g = torch.einsum("bknm,bmc->bknc", cheb_supports,
                           x)  # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,bnkio->bno', x_g,
                               weights) + bias  # b, N, dim_out

        return x_gconv

    def init_weights(self) -> None:
        torch.nn.init.orthogonal_(self.weights_pool.data)
        torch.nn.init.orthogonal_(self.bias_pool.data)


class ChebGCN(torch.nn.Module):
    """
    ChebNet with node_embeddings [node, dim]
    """

    def __init__(self, in_dim: int, out_dim: int, cheb_k: int, embed_dim: int, device: torch.device):
        super().__init__()
        self.device = device
        self.cheb_k = cheb_k
        # learned weights
        self.weights_pool = torch.nn.Parameter(
            torch.FloatTensor(embed_dim, cheb_k, in_dim, out_dim).to(self.device))
        self.bias_pool = torch.nn.Parameter(
            torch.FloatTensor(embed_dim, out_dim).to(self.device))

    def forward(self, x: torch.Tensor, node_embeddings: torch.Tensor) -> torch.Tensor:
        # node_embedding [n, d]  or dynamic [b, n, d]
        node_num = node_embeddings.shape[0]

        A = F.softmax(
            F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num, device=self.device), A]

        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(
                2 * A, support_set[-1]) - support_set[-2])
        cheb_supports = torch.stack(support_set, dim=0)
        # N, cheb_k, dim_in, dim_out
        weights = torch.einsum(
            'nd,dkio->nkio', node_embeddings, self.weights_pool)
        bias = torch.matmul(node_embeddings, self.bias_pool)  # N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", cheb_supports,
                           x)  # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g,
                               weights) + bias  # b, N, dim_out

        return x_gconv


class Time2Vec(torch.nn.Module):
    def __init__(self, time_dim: int, device: torch.device):
        super().__init__()

        self.device = device
        self.time_dim = time_dim
        self.w_0 = torch.nn.Parameter(
            torch.FloatTensor(1, 1).to(self.device), requires_grad=True)
        self.p_0 = torch.nn.Parameter(
            torch.FloatTensor(1, 1).to(self.device), requires_grad=True)
        self.W = torch.nn.Parameter(torch.FloatTensor(
            1, time_dim-1).to(self.device), requires_grad=True)
        self.P = torch.nn.Parameter(torch.FloatTensor(
            1, time_dim-1).to(self.device), requires_grad=True)
        self.F = torch.sin

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs [batch, times]
        return [batch, times, dim]
        """
        b_s = inputs.size(0)
        t_s = inputs.size(1)
        # import pdb; pdb.set_trace()
        v1 = self.F(torch.matmul(inputs.view(-1, t_s, 1), self.W) + self.P)
        v0 = torch.matmul(inputs.view(-1, t_s, 1), self.w_0) + self.p_0

        return torch.cat([v0, v1], -1)


class time_encoding(torch.nn.Module):
    ''' shift-invariant time encoding kernal

    inputs : [N, max_len]
    Returns: 3d float tensor which includes embedding dimension
    '''

    def __init__(self, time_dim: int, device: torch.device):
        super().__init__()

        self.device = device

        self.effe_numits = time_dim // 2
        self.time_dim = time_dim

        init_freq_base = np.linspace(0, 9, self.effe_numits).astype(np.float32)
        self.cos_freq_var = torch.nn.Parameter(
            torch.from_numpy(1 / 10.0 ** init_freq_base).to(device=self.device, dtype=torch.float32), requires_grad=False)
        self.sin_freq_var = torch.nn.Parameter(
            torch.from_numpy(1 / 10.0 ** init_freq_base).to(device=self.device, dtype=torch.float32), requires_grad=False)

        self.beta_var = torch.nn.Parameter(torch.from_numpy(np.ones(time_dim).astype(np.float32)).to(device=self.device, dtype=torch.float32), requires_grad=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        inputs = inputs.type(torch.FloatTensor).to(self.device)
        inputs = inputs.view(batch_size, seq_len, 1).repeat(
            1, 1, self.effe_numits)

        cos_feat = torch.sin(
            torch.mul(inputs, self.cos_freq_var.view(1, 1, self.effe_numits)))
        sin_feat = torch.cos(
            torch.mul(inputs, self.sin_freq_var.view(1, 1, self.effe_numits)))

        freq_feat = torch.cat((cos_feat, sin_feat), dim=-1)

        out = torch.mul(freq_feat, self.beta_var.view(1, 1, self.time_dim))
        return out


class TimeEncode(torch.nn.Module):
    '''
    2020 ICLR
    '''

    def __init__(self, time_dim: int, device: torch.device, factor: int = 5):
        super().__init__()
        self.time_dim = time_dim
        self.factor = factor
        self.device = device
        # (0,9) equally divided time_dim
        self.basis_freq = torch.nn.Parameter(torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim)).to(device=self.device, dtype=torch.float32))
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).to(device=self.device, dtype=torch.float32))

    def forward(self, ts: torch.Tensor) -> torch.Tensor:
        # ts: [batch, time, node]
        ts = ts.type(torch.FloatTensor).to(self.device)
        batch_size = ts.size(0)
        seq_len = ts.size(1)
        node_num = ts.size(2)
        ts = ts.view(batch_size, seq_len, node_num, 1)
        map_ts = ts * self.basis_freq.view(1, 1, 1, -1)
        map_ts += self.phase.view(1, 1, 1, -1)
        harmonic = torch.cos(map_ts)

        return harmonic


class GCGRU(torch.nn.Module):
    """
    RNNCell, 
    """

    def __init__(self, node_num: int, dim_in: int, dim_out: int, cheb_k: int, embed_dim: int, device: torch.device, time_with_station: bool = False, per_flag: bool = False):
        super().__init__()

        self.device = device

        self.node_num = node_num
        self.time_with_station = time_with_station
        self.hidden_dim = dim_out

        if time_with_station:
            self.gate = TGCN(dim_in+self.hidden_dim, 2*dim_out,
                             cheb_k, embed_dim, device=self.device, period=per_flag)  # 2*dim_out
            self.update = TGCN(dim_in+self.hidden_dim, dim_out,
                               cheb_k, embed_dim, device=self.device, period=per_flag)
        else:
            self.gate = ChebGCN(dim_in+self.hidden_dim, 2 *
                                dim_out, cheb_k, embed_dim, device=self.device)  # 2*dim_out
            self.update = ChebGCN(dim_in+self.hidden_dim,
                                  dim_out, cheb_k, embed_dim, device=self.device)

    def forward(self, x: torch.Tensor, state: torch.Tensor, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        DKGRNCell
        given kg do kgc, missing the periodic modeling, h_t
        1. $$h_t = z_t \odot h_{t-1} + (1-z_t) \odot \hat{h_t}$$ 
        2. $$\hat{h_t} = tanh(A[X_{:,t}, r \odot h_{t-1}]EW_{\hat{h}} + Eb_{\hat{h}})$$ 
        3. r_t = sigmoid(A[X_{:,t}, h_{t-1}]EW_r + Eb_r)
        4. z_t = sigmoid(A[X_{:,t}, h_{t-1}]EW_z + Eb_z)
        5. graph convolution
        """
        # x: B, num_nodes, input_dim(x, time_embedding)
        if self.time_with_station:
            node_embeddings, n_t, t, p = node_embeddings
        # state: B, num_nodes, hidden_dim : h_{t-1}
        # 1. [X:,t, h_{t-1}]
        # [b, n, dim_in_+dim_out]
        input_and_state = torch.cat((x, state), dim=-1)
        # 2. z_t   gate mechanism
        if self.time_with_station:
            # output dim: b, n, dim_out
            z_r = torch.sigmoid(
                self.gate(input_and_state, (node_embeddings, t, n_t, p)))
        else:
            z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        # z[b, n, dim_out]， r[b, n, dim_out]
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        # 3.
        candidate = torch.cat((x, r*state), dim=-1)
        # 5. update mechanism \hat{h_t}
        if self.time_with_station:
            _h = torch.tanh(self.update(
                candidate, (node_embeddings, t, n_t, p)))
        else:
            _h = torch.tanh(self.update(candidate, node_embeddings))
        # 6. final hidden state
        h = z*state + (1-z)*_h
        return h

    def init_hidden_state(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size, self.node_num, self.hidden_dim, device=self.device)


class GRUCell(torch.nn.Module):
    """
    RNNCell, 
    """

    def __init__(self, dim_in: int, dim_out: int, device: torch.device):
        super().__init__()
        self.device = device

        self.hidden_dim = dim_out
        self.W_z_r = torch.nn.Parameter(
            torch.FloatTensor(dim_in+dim_out, 2*dim_out).to(self.device))
        self.b_z_r = torch.nn.Parameter(torch.FloatTensor(2*dim_out).to(self.device))
        self.W_h = torch.nn.Parameter(
            torch.FloatTensor(dim_in+dim_out, dim_out).to(self.device))
        self.b_h = torch.nn.Parameter(torch.FloatTensor(dim_out).to(self.device))

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        DKGRNCell
        given kg do kgc, missing the periodic modeling, h_t
        1. $$h_t = z_t \odot h_{t-1} + (1-z_t) \odot \hat{h_t}$$ 
        2. $$\hat{h_t} = tanh(A[X_{:,t}, r \odot h_{t-1}]EW_{\hat{h}} + Eb_{\hat{h}})$$ 
        3. r_t = sigmoid([X_{:,t}, h_{t-1}]W_r + b_r)
        4. z_t = sigmoid([X_{:,t}, h_{t-1}]W_z + b_z)
        5. graph convolution
        """
        # x: B, num_nodes, input_dim(x, time_embedding)
        # state: B, num_nodes, hidden_dim : h_{t-1}
        # 1. [X:,t, h_{t-1}]
        # [b, n, dim_in_+dim_out]
        input_and_state = torch.cat((x, state), dim=-1)
        # 2. z_t   gate mechanism
        z_r = torch.einsum("io,bni->bno", self.W_z_r,
                           input_and_state) + self.b_z_r
        # z[b, n, dim_out]， r[b, n, dim_out]
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        # 3.
        candidate = torch.cat((x, r*state), dim=-1)
        # 5. update mechanism \hat{h_t}
        _h = torch.einsum("io,bni->bno", self.W_h, candidate) + self.b_h
        # 6. final hidden state
        h = z*state + (1-z)*_h
        return h

    def init_hidden_state(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size, self.node_num, self.hidden_dim, self.device)


class Encoder(torch.nn.Module):
    def __init__(self, node_num: int, in_dim: int, out_dim: int, cheb_k: int, embed_dim: int, device: torch.device, num_layers: int = 1, od_flag: bool = False, time_station: bool = False, g_d: str = "symm", per_flag: bool = False):
        super().__init__()
        assert num_layers >= 1, 'At least one DKGRNN layer in the Encoder.'

        self.device = device

        self.node_num = node_num
        self.input_dim = in_dim
        self.od_flag = od_flag
        self.g_d = g_d
        self.per_flag = per_flag
        self.time_station = time_station
        self.num_layers = num_layers
        self.gcgru_cells = torch.nn.ModuleList()

        self.gcgru_cells.append(
            GCGRU(node_num, in_dim, out_dim, cheb_k, embed_dim, self.device, time_station, per_flag))
        for _ in range(1, num_layers):
            self.gcgru_cells.append(
                GCGRU(node_num, out_dim, out_dim, cheb_k, embed_dim, self.device, time_station, per_flag))

    def forward(self, x: torch.Tensor, init_state: torch.Tensor, node_embeddings: torch.Tensor) -> typing.Tuple[torch.Tensor, typing.List[torch.Tensor]]:
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)

        if self.od_flag:
            od, do = node_embeddings

        if self.time_station:
            if self.g_d == "asym":
                node1, node2 = node_embeddings
            else:
                node_embeddings, t_embed = node_embeddings

        assert x.shape[2] == self.node_num
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []

        # tmp addding
        if self.per_flag:
            ps = []
            for dim_idx in range(x.shape[-1]):
                ps.append(torch.tanh(torch.bmm(x[..., dim_idx].transpose(1, 2), x[..., dim_idx])))
            p = torch.stack(ps, dim=-1).mean(dim=-1)
        else:
            p = None
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            # encoding
            for t in range(seq_length):

                if self.od_flag:
                    state = self.gcgru_cells[i](
                        current_inputs[:, t, :, :], state, (od[:, t, :, :], do[:, t, :, :]))
                elif self.time_station:
                    if self.g_d == 'asym':
                        state = self.gcgru_cells[i](
                            current_inputs[:, t, :, :], state, (node1[:, t, :, :], node2[:, t, :, :]))
                    else:
                        if t == 0:
                            state = self.gcgru_cells[i](
                                current_inputs[:, t, :, :], state, (node_embeddings, t_embed[:, 0, :], t_embed[:, 0, :], p))
                        else:
                            state = self.gcgru_cells[i](current_inputs[:, t, :, :], state, (
                                node_embeddings, t_embed[:, t, :], t_embed[:, t-1, :], p))
                else:
                    state = self.gcgru_cells[i](
                        current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            current_inputs = torch.stack(
                inner_states, dim=1)  # （B,T,N,Hidden_dim）
            output_hidden.append(state)  # (B, N, Hidden_dim)
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        init_states = []
        for i in range(self.num_layers):
            init_states.append(
                self.gcgru_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)


class Decoder(torch.nn.Module):
    # rewrite __init__ with types
    def __init__(self, node_num: int, horizon: int, in_dim: int, rnn_units: int, out_dim: int, cheb_k: int, embed_dim: int, device: torch.device, num_layers: int = 1, od_flag: bool = False, time_station: bool = False, g_d: str = "asym", per_flag: bool = False):
        super().__init__()
        assert num_layers >= 1, 'At least one DKGRNN layer in the Encoder.'

        self.device = device

        self.node_num = node_num
        self.input_dim = in_dim  # run_units plus time_embedding
        self.out_dim = out_dim  # output dim
        self.horizon = horizon
        self.num_layers = num_layers
        self.od_flag = od_flag
        self.g_d = g_d
        self.per_flag = per_flag
        self.time_station = time_station
        self.gcgru_cells = torch.nn.ModuleList()

        self.gcgru_cells.append(
            GCGRU(node_num, in_dim, rnn_units, cheb_k, embed_dim, self.device, time_station, per_flag))
        for _ in range(1, num_layers):
            self.gcgru_cells.append(
                GCGRU(node_num, rnn_units, rnn_units, cheb_k, embed_dim, self.device, time_station, per_flag))

        self.end_out = torch.nn.Linear(rnn_units, out_dim)
        self.mlp1 = torch.nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor, init_state: torch.Tensor, node_embeddings: torch.Tensor) -> torch.Tensor:
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)
        if self.time_station:
            if self.g_d == "asym":
                node1, node2 = node_embeddings
            else:
                node_embeddings, t_embed = node_embeddings

        assert x.shape[2] == self.node_num
        current_inputs = x
        input_len = x.shape[1]
        output_hidden = []

        # tmp addding
        if self.per_flag:
            tmp = self.mlp1(current_inputs).view(
                current_inputs.shape[0], current_inputs.shape[1], x.shape[2])
            p = torch.tanh(torch.bmm(tmp.transpose(1, 2), tmp))
        else:
            p = None

        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []

            for t in range(self.horizon):
                tin = t if t < input_len else input_len-1
                # node_embedding
                if self.time_station:
                    if self.g_d == "asym":
                        state = self.gcgru_cells[i](
                            current_inputs[:, tin, :, :], state, (node1[:, t, :, :], node2[:, t, :, :]))
                    else:
                        if t == 0:
                            state = self.gcgru_cells[i](
                                current_inputs[:, tin, :, :], state, (node_embeddings, t_embed[:, 0, :], t_embed[:, 0, :], p))
                        else:
                            state = self.gcgru_cells[i](current_inputs[:, tin, :, :], state, (
                                node_embeddings, t_embed[:, t, :], t_embed[:, t-1, :], p))
                else:
                    state = self.gcgru_cells[i](
                        current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            current_inputs = torch.stack(
                inner_states, dim=1)  # （B,T,N,Hidden_dim）
            output_hidden.append(state)  # (B, N, Hidden_dim)
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        out = self.end_out(current_inputs)
        return out

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        init_states = []
        for i in range(self.num_layers):
            init_states.append(
                self.gcgru_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)


class Decoder_WOG(torch.nn.Module):
    # rewrite__init__ with types
    def __init__(self, node_num: int, horizon: int, in_dim: int, rnn_units: int, out_dim: int, device: torch.device, num_layers: int = 1):
        super().__init__()
        assert num_layers >= 1, 'At least one DKGRNN layer in the Encoder.'

        self.device = device

        self.input_dim = in_dim  # run_units+time_embedding
        self.out_dim = out_dim  # output dim
        self.horizon = horizon
        self.node_num = node_num
        self.num_layers = num_layers
        self.gru_cells = torch.nn.ModuleList()

        # the input dim of first layer is different from the reminder layers
        self.gru_cells.append(GRUCell(in_dim, rnn_units, self.device))
        for _ in range(1, num_layers):
            self.gru_cells.append(GRUCell(rnn_units, rnn_units, self.device))

        self.end_out = torch.nn.Linear(rnn_units, out_dim, device=self.device)

    def forward(self, x: torch.Tensor, init_state: torch.Tensor) -> torch.Tensor:
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num
        current_inputs = x
        output_hidden = []

        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []

            for t in range(self.horizon):
                state = self.gru_cells[i](current_inputs[:, t, :, :], state)
                inner_states.append(state)
            current_inputs = torch.stack(
                inner_states, dim=1)  # （B,T,N,Hidden_dim）
            output_hidden.append(state)  # (B, N, Hidden_dim)
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        out = self.end_out(current_inputs)
        return out


class Model(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # random representation for node
        if self.graph_direction == "symm":
            self.node_embeddings = torch.nn.Parameter(torch.randn(
                self._n_vertex, self.embed_dim, device=self._device), requires_grad=True)
        if self.graph_direction == "asym":
            self.node_embeddings1 = torch.nn.Parameter(torch.randn(
                self._n_vertex, self.embed_dim, device=self._device), requires_grad=True)
            self.node_embeddings2 = torch.nn.Parameter(torch.randn(
                self._n_vertex, self.embed_dim, device=self._device), requires_grad=True)

        # embedding weight
        self.node_weight = torch.nn.Parameter(torch.randn(
            self.embed_dim, self.embed_dim, device=self._device), requires_grad=True)
        self.time_weight = torch.nn.Parameter(torch.randn(
            self.time_dim, self.time_dim, device=self._device), requires_grad=True)

        # time_embedding
        if self.time_embedding:
            # self.time_embed =torch.nn.Embedding(num_embeddings=self.num_time, embedding_dim=self.time_dim) # Embedding
            # self.time_embed = TimeEncode(self.time_dim) # 2020 ICLR
            self.time_embed = time_encoding(self.time_dim, device=self._device)  # 2019 NIPS
            # self.time_embed = Time2Vec(self.time_dim) # time2vec

        # node representation with time info
        if self.time_station:
            self.embed_dim += self.time_dim

        # encoder
        self.encoder = Encoder(self._n_vertex, self.x_dim, self.rnn_units, self.cheb_k,
                               self.embed_dim, self._device, self.num_layers, self.od_flag, self.time_station, self.graph_direction, self.period)

        # decoder
        if self.Seq_Dec:
            if self.time_embedding:
                # self.x_dim = self.rnn_units + self.time_dim
                self.x_dim = self.rnn_units
            else:
                self.x_dim = self.rnn_units

            if self.od_flag:
                self.decoder = Decoder_WOG(self._n_vertex, self.y_len, self.x_dim, self.rnn_units, self.y_dim, self._device, self.num_layers)
            else:
                self.decoder = Decoder(self._n_vertex, self.y_len, self.x_dim, self.rnn_units, self.y_dim,  self.cheb_k,
                                       self.embed_dim, self._device, self.num_layers, self.od_flag, self.time_station, self.graph_direction, self.period)

        else:  # 一次解码
            self.decoder = torch.nn.Conv2d(
                1, self.y_len * self.y_dim, kernel_size=(1, self.rnn_units), bias=True, device=self._device)

    def forward(self, x: torch.Tensor, additional_info: typing.Dict[str, typing.Any]) -> torch.Tensor:
        if self.time_embedding and self.constrative_time and self.training:
            x_time: int = additional_info['x_time']\
                if 'x_time' in additional_info else None
            y_time: int = additional_info['y_time']\
                if 'y_time' in additional_info else None
            cons_time: int = additional_info['cons_time']\
                 if 'cons_time' in additional_info else None
        elif self.time_embedding:
            if self.od_flag:
                x_time: int = additional_info['x_time']\
                    if 'x_time' in additional_info else None
                y_time: int = additional_info['y_time']\
                    if 'y_time' in additional_info else None
                od: int = additional_info['od']\
                    if 'od' in additional_info else None
                do: int = additional_info['do']\
                    if 'do' in additional_info else None
            else:
                x_time: int = additional_info['x_time']\
                    if 'x_time' in additional_info else None
                y_time: int = additional_info['y_time']\
                    if 'y_time' in additional_info else None
        else:
            pass

        """
        if time_embedding is true:
            input: X [batch, T, node, feat], X_time [batch, T], Y_time [batch, T]
        else:
            input: X
        """

        if self.time_embedding:
            # [batch, time, dim]
            x_time_feat = self.time_embed(x_time)
            y_time_feat = self.time_embed(y_time)

        # add node embedding constrains.
        node = F.tanh(torch.mm(self.node_embeddings, self.node_weight))

        # init hidden states
        init_state = self.encoder.init_hidden(x.shape[0])

        # get the last hidden states from the last layer and every layer
        if self.od_flag:
            # output: [B, T, N, hidden]
            output, _ = self.encoder(x, init_state, (od, do))

        elif self.time_station:
            # graph direction
            if self.graph_direction == "symm":
                # change <[e1||t], [e2||t]> --> <e1, e2>+<t, t-1>
                # _node_embed = self.stat_with_time(self.node_embeddings, x_time_feat)
                output, _ = self.encoder(
                    x, init_state, (node, x_time_feat))
            if self.graph_direction == "asym":
                node1 = self.stat_with_time(self.node_embeddings1, x_time_feat)
                node2 = self.stat_with_time(self.node_embeddings2, x_time_feat)
                output, _ = self.encoder(x, init_state, (node1, node2))
        else:
            # output: [B, T, N, hidden]
            output, _ = self.encoder(x, init_state, self.node_embeddings)

        # decoding
        if self.Seq_Dec:
            init_state = self.decoder.init_hidden(x.shape[0])

            if self.od_flag:
                outs = self.decoder(output, _)

            if self.time_station:
                if self.graph_direction == "symm":
                    # _node_embed = self.stat_with_time(self.node_embeddings, x_time_feat)
                    outs = self.decoder(output, _, (node, y_time_feat))
                if self.graph_direction == "asym":
                    node1 = self.stat_with_time(
                        self.node_embeddings1, y_time_feat)
                    node2 = self.stat_with_time(
                        self.node_embeddings2, y_time_feat)
                    outs = self.decoder(output, _, (node1, node2))
            else:
                outs = self.decoder(output, _, self.node_embeddings)
        else:
            output = output[:, -1:, :, :]  # B, 1, N, hidden
            # CNN based predictor
            output = self.decoder((output))  # B, T*C, N, 1
            output = output.squeeze(-1).reshape(-1,
                                                self.y_len, self.y_dim, self._n_vertex)
            outs = output.permute(0, 1, 3, 2)  # B, T, N, C

        if self.constrative_time and self.training:
            cons_time_embed = self.time_embed(cons_time)
            ratios = []

            for i in range(1, 4):
                embed_diff = torch.mean(
                    (cons_time_embed[:, 0, :] - cons_time_embed[:, i, :]) ** 2, dim=-1, keepdim=True)  # 16 x 1
                distance_diff = torch.abs(
                    cons_time[:, 0] - cons_time[:, i])  # 16 x 1
                ratio = embed_diff / distance_diff.float().clamp_(1e-6)
                ratios.append(ratio)
            return outs, ratios

        return outs

    def stat_with_time(self, node_embeddings: torch.Tensor, time_feat: torch.Tensor) -> torch.Tensor:
        # node_embedding [n, dim], time [batch, T, dim] -> [batch, T, N, dim]
        _node_embed = node_embeddings.repeat(
            time_feat.size(0), time_feat.size(1), 1, 1)
        _node_embed = torch.cat((_node_embed, time_feat), dim=-1)

        return _node_embed
