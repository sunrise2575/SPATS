import math

import torch
import typing

from .__base import ModelBase


class Model(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fan_in = self._n_vertex * self.x_dim

        self.phi = torch.nn.Parameter(torch.zeros(
            self.x_len, self.fan_in, device=self._device))
        self.theta = torch.nn.Parameter(torch.zeros(
            self.y_len, self.fan_in, device=self._device))
        self.bias = torch.nn.Parameter(
            torch.zeros(self.fan_in, device=self._device))

        self._reset_parameters()

        # ARMA(p, q)
        # x_t = c + A_1 * x_{t-1} + ... + A_p * x_{t-p} + e_t + B_1 * e_{t-1} + ... + B_q * e_{t-q}
        # A_1, ..., A_p are phi
        # B_1, ..., B_q are theta
        # c is considered as self.bias
        # e_t is considered as error (recursively calculated)
        # I set variables as:
        # p = x_len
        # q = y_len

    def _reset_parameters(self) -> None:
        # affected from pytorch torch.nn.Linear()
        torch.nn.init.kaiming_uniform_(self.phi, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.theta, a=math.sqrt(5))

        if self.bias is not None:
            bound = 1 / math.sqrt(self.fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, additional_info: typing.Dict[str, typing.Any]) -> torch.Tensor:
        y: torch.Tensor = additional_info['y']\
            if 'y' in additional_info else None
        B, x_len, V, x_dim = x.shape
        B, y_len, V, _ = y.shape

        x = x.view(B, x_len, V * x_dim)
        # x = [B, x_len, V*x_dim]
        y = y.view(B, y_len, V * x_dim)
        # y = [B, y_len, V*x_dim]

        y_hats = []
        errors = []

        for i in range(y_len):
            # AR (auto regressive) model
            y_hat = torch.einsum('jk,ijk->ik', self.phi, x)
            # y_hat = [B, V*x_dim]

            if len(errors) > 0:
                # MA (moving average) model
                errors_stacked = torch.stack(errors, dim=1)
                # errors_stacked = [B, len(errors), V*x_dim]

                error_mat = torch.cat([
                    errors_stacked,
                    torch.zeros(B, y_len - len(errors),
                                V * x_dim).to(x.device),
                ], dim=1)
                # error_mat = [B, y_len, V*x_dim]

                weighted_error = torch.einsum(
                    'jk,ijk->ik', self.theta, error_mat)
                # weighted_error = [B, V*x_dim]

                y_hats.append(y_hat + weighted_error + self.bias)

            else:
                y_hats.append(y_hat + self.bias)

            # prepare moving average error for next iteration
            error = y[:, i, :] - y_hat
            # error = [B, V*x_dim]
            errors.append(error)

        result = torch.stack(y_hats, dim=1)
        # result = [B, y_len, V*x_dim]
        result = result.view(B, y_len, V, x_dim)
        # result = [B, y_len, V, x_dim]

        return result
