import typing

import torch
import torch.nn.functional

from .__base import ModelBase

# import torch.cuda.nvtx as nvtx


class Spatial_Attention_layer(torch.nn.Module):
    '''
    compute spatial attention scores
    '''

    def __init__(self, x_dim, num_of_vertices, num_of_timesteps, device: torch.device = None):
        super().__init__()
        self.W1 = torch.nn.Parameter(
            torch.FloatTensor(num_of_timesteps).to(device))
        self.W2 = torch.nn.Parameter(torch.FloatTensor(
            x_dim, num_of_timesteps).to(device))
        self.W3 = torch.nn.Parameter(torch.FloatTensor(x_dim).to(device))
        self.bs = torch.nn.Parameter(torch.FloatTensor(
            1, num_of_vertices, num_of_vertices).to(device))
        self.Vs = torch.nn.Parameter(torch.FloatTensor(
            num_of_vertices, num_of_vertices).to(device))
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # nvtx.range_push(f'{self.__class__.__name__}')
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B,N,N)
        '''
        # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)

        # (F)(b,N,F,T)->(b,N,T)->(b,T,N)
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)

        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)

        S = torch.matmul(self.Vs, torch.sigmoid(
            product + self.bs))  # (N,N)(B, N, N)->(B,N,N)

        # S_normalized = torch.nn.functional.softmax(S, dim=1)
        S_normalized = self.softmax(S)
        # nvtx.range_pop()

        return S_normalized


class cheb_conv_withSAt(torch.nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, cheb_polynomials: typing.List[torch.Tensor],
                 x_dim: int, out_channels: int, device: torch.device = None):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super().__init__()
        self.K = len(cheb_polynomials)
        self.cheb_polynomials = cheb_polynomials
        self.x_dim = x_dim
        self.out_channels = out_channels
        self.Theta = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(
            x_dim, out_channels).to(device)) for _ in range(self.K)])

    def forward(self, x: torch.Tensor, spatial_attention: torch.Tensor) -> torch.Tensor:
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''
        # nvtx.range_push(f'{self.__class__.__name__}')
        # batch_size, num_of_vertices, _, num_of_timesteps = x.shape
        _, _, _, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            # output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to('cuda:0')  # (b, N, F_out)
            output = None

            for k in range(self.K):
                T_k = self.cheb_polynomials[k]  # (N,N)

                # (N,N)*(N,N) = (N,N) 多行和为1, 按着列进行归一化
                # nvtx.range_push(f'mul(spatial_attention)')
                T_k_with_at = T_k.mul(spatial_attention)
                # nvtx.range_pop()

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                # nvtx.range_push(f'permute().matmul()')
                # (N, N)(b, N, F_in) = (b, N, F_in) 因为是左乘，所以多行和为1变为多列和为1，即一行之和为1，进行左乘
                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)
                # nvtx.range_pop()

                # (b, N, F_in)(F_in, F_out) = (b, N, F_out)
                # output = output + rhs.matmul(theta_k)
                if output is None:
                    output = rhs.matmul(theta_k)
                else:
                    output += rhs.matmul(theta_k)

            # nvtx.range_push(f'output.unsqueeze()')
            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)
            # nvtx.range_pop()

        # nvtx.range_pop()
        # (b, N, F_out, T)
        return torch.nn.functional.relu(torch.cat(outputs, dim=-1))


class Temporal_Attention_layer(torch.nn.Module):
    def __init__(self, x_dim: int, num_of_vertices: int, num_of_timesteps: int, device: torch.device = None):
        super().__init__()
        self.U1 = torch.nn.Parameter(
            torch.FloatTensor(num_of_vertices).to(device))
        self.U2 = torch.nn.Parameter(
            torch.FloatTensor(x_dim, num_of_vertices).to(device))
        self.U3 = torch.nn.Parameter(torch.FloatTensor(x_dim).to(device))
        self.be = torch.nn.Parameter(torch.FloatTensor(
            1, num_of_timesteps, num_of_timesteps).to(device))
        self.Ve = torch.nn.Parameter(torch.FloatTensor(
            num_of_timesteps, num_of_timesteps).to(device))
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        '''
        # nvtx.range_push(f'{self.__class__.__name__}')
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        lhs = torch.matmul(torch.matmul(
            x.permute(0, 3, 2, 1), self.U1), self.U2)
        # print(f'{self.U1.shape=}')
        # print(f'{self.U1[:10]=}')
        # print(f'{self.U2.shape=}')
        # print(f'{self.U2[0, :10]=}')
        # print(f'{lhs.shape=}')
        # print(f'{lhs[0, :, 0]=}')
        # x:(B, N, F_in, T) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)

        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)
        # print(f'{rhs.shape=}')
        # print(f'{rhs[0, 0, :]=}')

        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)
        # print(f'{product.shape=}')
        # print(f'{product[0, :10, :10]=}')

        E = torch.matmul(self.Ve, torch.sigmoid(
            product + self.be))  # (B, T, T)
        # print(f'{E.shape=}')
        # print(f'{E[0, :10, :10]=}')

        # E_normalized = torch.nn.functional.softmax(E, dim=1)
        E_normalized = self.softmax(E)
        # print(f'{E_normalized.shape=}')
        # print(f'{E_normalized[0, :10, :10]=}')

        # nvtx.range_pop()

        return E_normalized


class cheb_conv(torch.nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, cheb_polynomials: typing.List[torch.Tensor],
                 x_dim: int, out_channels: int, device: torch.device = None):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super().__init__()
        self.K = len(cheb_polynomials)
        self.cheb_polynomials = cheb_polynomials
        self.x_dim = x_dim
        self.out_channels = out_channels
        self.Theta = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor(
            x_dim, out_channels).to(device)) for _ in range(self.K)])
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''
        # batch_size, num_of_vertices, _, num_of_timesteps = x.shape
        # nvtx.range_push(f'{self.__class__.__name__}')
        _, _, _, num_of_timesteps = x.shape
        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            # output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to('cuda:0')  # (b, N, F_out)
            output = None

            for k in range(self.K):
                T_k = self.cheb_polynomials[k]  # (N,N)
                theta_k = self.Theta[k]  # (in_channel, out_channel)
                rhs = graph_signal.permute(
                    0, 2, 1).matmul(T_k).permute(0, 2, 1)

                # output = output + rhs.matmul(theta_k)
                if output is None:
                    output = rhs.matmul(theta_k)
                else:
                    output += rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        # nvtx.range_pop()

        # return torch.nn.functional.relu(torch.cat(outputs, dim=-1))
        return self.relu(torch.cat(outputs, dim=-1))


class ASTGCN_block(torch.nn.Module):
    def __init__(self, x_dim: int, nb_chev_filter: int, nb_time_filter: int, time_strides: int,
                 cheb_polynomials: typing.List[torch.Tensor],
                 num_of_vertices: int, num_of_timesteps: int, device: torch.device = None):
        super().__init__()

        self.TAt = Temporal_Attention_layer(
            x_dim, num_of_vertices, num_of_timesteps, device=device)
        self.SAt = Spatial_Attention_layer(
            x_dim, num_of_vertices, num_of_timesteps, device=device)
        self.cheb_conv_SAt = cheb_conv_withSAt(
            cheb_polynomials, x_dim, nb_chev_filter, device=device)
        self.time_conv = torch.nn.Conv2d(
            nb_chev_filter, nb_time_filter,
            kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))
        self.residual_conv = torch.nn.Conv2d(
            x_dim, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = torch.nn.LayerNorm(nb_time_filter)  # 需要将channel放到最后一个维度上

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # nvtx.range_push(f'{self.__class__.__name__}')
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, nb_time_filter, T)
        '''
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        # print(f'{x.shape=}')
        # print(f'{x[0, 0, 0, :]=}')

        # TAt
        temporal_At = self.TAt(x)  # (b, T, T)
        # print(f'{temporal_At.shape=}')
        # print(f'{temporal_At[0, :10, :10]=}')

        x_TAt = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps), temporal_At).reshape(
            batch_size, num_of_vertices, num_of_features, num_of_timesteps)
        # print(f'{x_TAt.shape=}')
        # print(f'{x_TAt[0, 0, 0, :]=}')

        # SAt
        spatial_At = self.SAt(x_TAt)
        # print(f'{spatial_At.shape=}')
        # print(f'{spatial_At[0, :10, :10]=}')

        # cheb gcn
        spatial_gcn = self.cheb_conv_SAt(x, spatial_At)  # b,N,F,T)
        # print(f'{spatial_gcn.shape=}')
        # print(f'{spatial_gcn[0, 0, 0, :]=}')
        # spatial_gcn = self.cheb_conv(x)

        # convolution along the time axis
        # (b,N,F,T)->(b,F,N,T) 用(1,3)的卷积核去做->(b,F,N,T)
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))
        # print(f'{time_conv_output.shape=}')
        # print(f'{time_conv_output[0, 0, 0, :]=}')

        # residual shortcut
        # (b,N,F,T)->(b,F,N,T) 用(1,1)的卷积核去做->(b,F,N,T)
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))

        x_residual = self.ln(
            torch.nn.functional.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        # (b,F,N,T)->(b,T,N,F) -ln-> (b,T,N,F)->(b,N,F,T)

        # nvtx.range_pop()

        return x_residual


class Model(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.BlockList = torch.nn.ModuleList([
            ASTGCN_block(int(self.x_dim),
                         int(self.nb_chev_filter),
                         int(self.nb_time_filter),
                         int(self.time_strides),
                         self._supports,
                         int(self._n_vertex),
                         int(self.x_len),
                         device=self._device)])

        self.BlockList.extend([
            ASTGCN_block(int(self.nb_time_filter),
                         int(self.nb_chev_filter),
                         int(self.nb_time_filter),
                         1,
                         self._supports,
                         int(self._n_vertex),
                         int(self.x_len // self.time_strides),
                         device=self._device) for _ in range(self.nb_block - 1)])

        self.final_conv = torch.nn.Conv2d(
            int(self.x_len / self.time_strides),
            int(self.y_len),
            kernel_size=(1, self.nb_time_filter))

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.uniform_(p)

    def forward(self, x: torch.Tensor, *_) -> torch.Tensor:
        # nvtx.range_push(f'{self.__class__.__name__}')
        x = x.permute((0, 2, 3, 1))
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''

        for _, block in enumerate(self.BlockList):
            x = block(x)

        output = x.permute(0, 3, 1, 2)

        output = self.final_conv(output)

        # output = output[:, :, :, -1]
        # output = output.permute(0, 2, 1)

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)

        # nvtx.range_pop()

        return output
