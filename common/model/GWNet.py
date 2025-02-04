import math
import typing

import torch
import torch.nn.functional

from .__base import ModelBase


class nconv(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class linear(torch.nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.mlp = torch.torch.nn.Conv2d(c_in, c_out, kernel_size=(
            1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class gcn(torch.nn.Module):
    def __init__(self, c_in: int, c_out: int, dropout: float, support_len: int = 3, order: int = 2):
        super().__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x: torch.Tensor, support: typing.List[torch.Tensor]) -> torch.Tensor:
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = torch.nn.functional.dropout(
            h, p=self.dropout, training=self.training)
        return h


class Model(ModelBase):
    # supports means graph
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.blocks = int((self.y_dim + 1) / 3)  # duct taping

        self.filter_convs = torch.nn.ModuleList()
        self.gate_convs = torch.nn.ModuleList()
        self.residual_convs = torch.nn.ModuleList()
        self.skip_convs = torch.nn.ModuleList()
        self.bn = torch.nn.ModuleList()
        self.gconv = torch.nn.ModuleList()

        self.start_conv = torch.nn.Conv2d(in_channels=self.x_dim,
                                          out_channels=self.residual_channels,
                                          kernel_size=(1, 1))

        receptive_field = 1

        self.layers, self.blocks = self._select_layers_and_blocks()

        self.aptinit = None

        self._supports_len = 0
        if self._supports is not None:
            self._supports_len += len(self._supports)

        if self.gcn_bool and self.addaptadj:
            if self.aptinit is None:
                if self._supports is None:
                    self._supports = []
                self.nodevec1 = torch.nn.Parameter(
                    torch.randn(self._n_vertex, 10), requires_grad=True)
                self.nodevec2 = torch.nn.Parameter(
                    torch.randn(10, self._n_vertex), requires_grad=True)
                self._supports_len += 1
            else:
                if self._supports is None:
                    self._supports = []
                m, p, n = torch.svd(self.aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = torch.nn.Parameter(
                    initemb1, requires_grad=True)
                self.nodevec2 = torch.nn.Parameter(
                    initemb2, requires_grad=True)
                self._supports_len += 1

        # Conv1d not working in pytorch 1.12
        # I modified all Conv1d to Conv2d
        for _ in range(self.blocks):
            additional_scope = self.kernel_size - 1
            new_dilation = 1
            for _ in range(self.layers):
                # dilated convolutions
                self.filter_convs.append(torch.nn.Conv2d(in_channels=self.residual_channels,
                                                         out_channels=self.dilation_channels,
                                                         kernel_size=(1, self.kernel_size), dilation=new_dilation))

                self.gate_convs.append(torch.nn.Conv2d(in_channels=self.residual_channels,
                                                       out_channels=self.dilation_channels,
                                                       kernel_size=(1, self.kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(torch.nn.Conv2d(in_channels=self.dilation_channels,
                                                           out_channels=self.residual_channels,
                                                           kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(torch.nn.Conv2d(in_channels=self.dilation_channels,
                                                       out_channels=self.skip_channels,
                                                       kernel_size=(1, 1)))
                self.bn.append(torch.nn.BatchNorm2d(self.residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(
                        self.dilation_channels, self.residual_channels, self.dropout, support_len=self._supports_len))

        self.end_conv_1 = torch.nn.Conv2d(in_channels=self.skip_channels,
                                          out_channels=self.end_channels,
                                          kernel_size=(1, 1),
                                          bias=True)

        # in the original Graph WaveNet code:
        #   seq_length==out_dim==12 means y_len
        #   in_dim==2 means x_dim
        # so the out_channels of self.end_conv_2 is self.y_len
        self.end_conv_2 = torch.nn.Conv2d(in_channels=self.end_channels,
                                          out_channels=self.y_len,
                                          kernel_size=(1, 1),
                                          bias=True)

        self.receptive_field = receptive_field

    def forward(self, input: torch.Tensor, *_) -> torch.Tensor:
        # input: (B, x_len, V, x_dim)
        input = input.transpose(1, 3)
        # input: (B, x_dim, V, x_len)

        if self.x_len < self.receptive_field:
            # time-offset for dilated convolution
            # because dilated convolution's receptive field is:
            # (2^layers - 1) * blocks + 1;
            # so in the case of x_len=12, layer=2 and blocks=4,
            # the receptive field size is (2^2 - 1) * 4 + 1 = 13 > x_len=12
            # therefore add single 0s to x's left(=oldest side); which is padding (1, 0, 0, 0)
            x = torch.nn.functional.pad(
                input, (self.receptive_field - self.x_len, 0, 0, 0))
            # x: (B, x_dim, V, x_len + receptive_field - x_len)
        else:
            x = input
            # x: (B, x_dim, V, x_len)

        x = self.start_conv(x)
        # x: (B, self.residual_channels, V, x_len)

        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self._supports is not None:
            # self.nodevec1: (V, 10)
            # self.nodevec2: (10, V)
            # adp: (V, V)
            adp = torch.nn.functional.softmax(torch.nn.functional.relu(
                torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            # self._supports = [forward, backward]
            new_supports = self._supports + [adp]
            # new_supports = [forward, backward, adaptive]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            residual = x
            # x and residual is getting shorter
            # residual: (B, residual_channels, V, current_x_len)

            # Note:

            # dilated convolution
            # self.filter_convs is kernel_size=(1,2), therefore:
            #   (B, C_in, H_in, W_in) -> (B, C_out, H_in, W_in - dilation)
            filter = self.filter_convs[i](residual)
            # filter: (B, self.dilation_channels, V, current_x_len - (layer's dilation))
            filter = torch.tanh(filter)

            # self.gate_convs is kernel_size=(1,2), therefore:
            #   (B, C_in, H_in, W_in) -> (B, C_out, H_in, W_in - dilation)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            # gate: (B, self.dilation_channels, V, current_x_len - (layer's dilation))

            x = filter * gate
            # x: (B, self.dilation_channels, V, current_x_len - (layer's dilation))
            # Note: operator '*' means element-wise multiplication (Hadamard product)

            # parametrized skip connection
            s = x
            # s: (B, self.dilation_channels, V, current_x_len - (layer's dilation))
            s = self.skip_convs[i](s)
            # s: (B, self.skip_channels, V, current_x_len - (layer's dilation))
            try:
                # skip: (B, self.skip_channels, V, before_len)
                skip = skip[:, :, :, -s.size(3):]
                # skip: (B, self.skip_channels, V, current_x_len - (layer's dilation)); skip becomes shorter!
                # or skip = 0
            except:
                skip = 0
            skip = s + skip
            # if skip is = 0, then skip will be s: (B, self.skip_channels, V, current_x_len - (layer's dilation))

            if self.gcn_bool and self._supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                    # x: (B, self.residual_channels, V, current_x_len - (layer's dilation))
                else:
                    x = self.gconv[i](x, self._supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]

            # batch normalization
            x = self.bn[i](x)

        # skip: (B, self.skip_channels, V, 1)
        x = torch.nn.functional.relu(skip)
        # x: (B, self.skip_channels, V, 1)
        x = torch.nn.functional.relu(self.end_conv_1(x))
        # x: (B, self.end_channels, V, 1)
        x = self.end_conv_2(x)
        # x: (B, self.y_len, V, 1)

        return x

    def _select_layers_and_blocks(self) -> typing.Tuple[int, int]:
        # this function selects the suitable number of layers and blocks
        #     not from the original GWNet code; written by me

        # total receptive field is by adding all the dilations and add one;
        #     in the case of layer=4 and blocks=3,
        #     the receptive field size is (1+2+4+8+1+2+4+8+1+2+4+8)+1=46
        # receptive field is calculated as (2^layers - 1) * blocks + 1;
        #     in the case of layer=4 and blocks=3,
        #     the receptive field size is (2^4 - 1) * 3 + 1 = 46

        # Quick lookup table for receptive field size
        # row: layers, column: blocks
        #      1    2    3    4    5    6    7    8
        # 1    2    3    4    5    6    7    8    9
        # 2    4    7   10   13   16   19   22   25
        # 3    8   15   22   29   36   43   50   57
        # 4   16   31   46   61   76   91  106  121
        # 5   32   63   94  125  156  187  218  249
        # 6   64  127  190  253  316  379  442  505
        # 7  128  255  382  509  636  763  890 1017
        # 8  256  511  766 1021 1276 1531 1786 2041
        # 9  512 1023 1534 2045 2556 3067 3578 4089

        r = self.x_len

        if r in [1, 2]:
            # if x_len is 1 or 2, there are only one best option
            return 1, 1

        candidates = []

        L_max = math.ceil(math.log2(r))  # maximum available layers
        B_max = r-1  # maximum available blocks

        for L in range(1, L_max+1):
            # minimum available blocks
            B_min = math.floor((r - 1)/(2 ** L - 1))
            if B_min == 0:
                break

            for B in range(B_min, B_max+1):
                r_hat = (2 ** L - 1) * B + 1  # receptive field size
                if r_hat >= r:
                    cost = r_hat + L * B  # cost function; L*B = number of Conv2d
                    candidates.append((L, B, cost, L*B))
                    break

        # sort by cost and then by number of Conv2d if cost is same. select the top candidate
        candidate = sorted(candidates, key=lambda x: (x[2], x[3]))[0]

        # return the selected layers and blocks
        return candidate[0], candidate[1]
