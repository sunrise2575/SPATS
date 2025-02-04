import math

import torch

from .__base import ModelBase


class Model(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_params = self._n_vertex * self.x_dim

        # 3D weight!
        self.weight = torch.nn.Parameter(
            torch.empty(self.x_len, self.num_params, self.num_params, device=self._device))

        self.bias = torch.nn.Parameter(
            torch.empty(self.num_params, device=self._device))

        self._reset_parameters()

        # VAR(p)
        # x_t = c + A_1 * x_{t-1} + ... + A_p * x_{t-p} + e_t
        # A_1, ..., A_p are weights
        # self.weights is modulelist like [A_1, ..., A_p]
        # c and e_t is considered as self.bias

    def _reset_parameters(self) -> None:
        # from pytorch torch.nn.Linear()
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in = self.num_params
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, *_) -> torch.Tensor:
        # x = [B, x_len, V, x_dim]
        B, x_len, V, x_dim = x.shape

        x = x.view(B, x_len, V * x_dim)
        # x = [B, x_len, V * x_dim]

        model_outputs = []
        cur_model_input = x

        for _ in range(self.y_len):
            # this einsum is exactly same as: x_t = A_1 * x_{t-1} + ... + A_p * x_{t-p}
            # b = batch
            # t = time
            # i, j = self.num_params
            _model_output = torch.einsum(
                'tij,btj->bi', self.weight, cur_model_input)
            # out = [B, V * x_dim]

            # self.bias = [V * x_dim]
            _model_output = _model_output + self.bias

            # _model_output = [B, V, x_dim]
            model_outputs.append(_model_output)

            _model_output = _model_output.unsqueeze(dim=1)

            # _model_output = [B, 1, V * x_dim]
            # cur_model_input[:, 1:, ...] = [B, x_len - 1, V * x_dim]
            next_input = torch.cat(
                [cur_model_input[:, 1:, ...], _model_output], axis=1)
            # next_input = [B, x_len, V * x_dim]
            cur_model_input = next_input

        result: torch.Tensor = torch.stack(model_outputs, axis=1)
        # result = [B, self.y_len, V * x_dim]

        result = result.view(B, self.y_len, V, x_dim)

        return result
