import typing

import numpy as np
import torch

from .__base import ModelBase


class LayerParams:
    def __init__(self, rnn_network: torch.nn.Module, layer_type: str, device: torch.device = None):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type
        self._device = device

    def get_weights(self, shape: typing.Tuple[int, ...]) -> torch.nn.Parameter:
        if shape not in self._params_dict:
            nn_param = torch.nn.Parameter(
                torch.empty(*shape, device=self._device))
            torch.nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter(
                '{}_weight_{}'.format(self._type, str(shape)),
                nn_param)
        return self._params_dict[shape]

    def get_biases(self, length: int, bias_start=0.0) -> torch.nn.Parameter:
        if length not in self._biases_dict:
            biases = torch.nn.Parameter(
                torch.empty(length, device=self._device))
            torch.nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter(
                '{}_biases_{}'.format(self._type, str(length)),
                biases)
        return self._biases_dict[length]


class DCGRUCell(torch.nn.Module):
    def __init__(self, arg: dict, support: typing.List[torch.Tensor]):
        super().__init__()

        # dictinary value to class member variable
        for k, v in arg.items():
            setattr(self, k, v)

        # self._activation = torch.relu
        self._activation = torch.tanh
        self._num_nodes = support[0].shape[0]
        self._num_units = self.rnn_units

        self._supports = support

        self._gconv_params = LayerParams(
            rnn_network=self,
            layer_type='gconv',
            device=support[0].device)

    def forward(self, inputs: torch.Tensor, hx: torch.Tensor) -> torch.Tensor:
        output_size = 2 * self._num_units
        value = torch.sigmoid(self._gconv(
            inputs, hx, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(
            tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

        c = self._gconv(inputs, r * hx, self._num_units)
        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state

    def _concat(self, x: torch.Tensor, x_: torch.Tensor) -> torch.Tensor:
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def _gconv(self, inputs: torch.Tensor, state: torch.Tensor, output_size: torch.Tensor, bias_start: float = 0.0) -> torch.Tensor:
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(
            x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        if self.max_diffusion_step > 0:
            for support in self._supports:
                x1 = torch.sparse.mm(support, x0)
                x = self._concat(x, x1)

                for _ in range(2, self.max_diffusion_step + 1):
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        # same as: len(self._supports) * self._max_diffusion_step + 1
        num_matrices = x.shape[0]
        x = torch.reshape(
            x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(
            x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

        weights = self._gconv_params.get_weights(
            (input_size * num_matrices, output_size))
        # (batch_size * self._num_nodes, output_size)
        x = torch.matmul(x, weights)

        biases = self._gconv_params.get_biases(output_size, bias_start)
        x += biases
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])


class EncoderModel(torch.nn.Module):
    def __init__(self, arg: dict, support: typing.List[np.ndarray]):
        super().__init__()

        # dictinary value to class member variable
        for k, v in arg.items():
            setattr(self, k, v)

        self.num_vertices = support[0].shape[0]
        self.hidden_state_size = self.num_vertices * self.rnn_units
        self.dcgru_layers = torch.nn.ModuleList([
            DCGRUCell(arg, support) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs: torch.Tensor, hidden_state: torch.Tensor = None) \
            -> typing.Tuple[torch.Tensor, torch.Tensor]:
        batch_size, _ = inputs.size()

        if hidden_state is None:
            hidden_state = torch.zeros(
                (len(self.dcgru_layers), batch_size, self.hidden_state_size),
                device=inputs.device)

        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        # runs in O(num_layers) so not too slow
        return output, torch.stack(hidden_states)


class DecoderModel(torch.nn.Module):
    def __init__(self, arg: dict, support: typing.List[np.ndarray]):
        super().__init__()

        # dictinary value to class member variable
        for k, v in arg.items():
            setattr(self, k, v)

        self.num_vertices = support[0].shape[0]
        # self.arg = arg
        # self.hidden_state_size = self.num_vertices * self.rnn_units
        self.projection_layer = torch.nn.Linear(self.rnn_units, self.y_dim)
        self.dcgru_layers = torch.nn.ModuleList([
            DCGRUCell(arg, support) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs: torch.Tensor, hidden_state: torch.Tensor = None) \
            -> typing.Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_vertices * self.y_dim)
        return output, torch.stack(hidden_states)


class Model(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # convert supports to typing.List[torch.sparse.Tensor] format:
        my_supports = []
        for support in self._supports:
            indices = torch.nonzero(support).T
            values = support[indices[0], indices[1]]
            support_sparse = torch.sparse_coo_tensor(
                indices, values, support.shape)
            my_supports.append(support_sparse)

        self.encoder_model = EncoderModel(self._arg, my_supports)
        self.decoder_model = DecoderModel(self._arg, my_supports)

    def _compute_sampling_threshold(self, batches_seen: int) -> float:
        return self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(float(batches_seen) / self.cl_decay_steps))

    def _encoder(self, inputs: torch.Tensor) -> torch.Tensor:
        encoder_hidden_state = None
        for t in range(inputs.shape[0]):
            _, encoder_hidden_state = self.encoder_model(
                inputs[t], encoder_hidden_state)

        return encoder_hidden_state

    def _decoder(self, encoder_hidden_state: torch.Tensor, labels: torch.Tensor = None, batches_seen: int = None) -> torch.Tensor:
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros(
            (batch_size, self._n_vertex * self.y_dim),
            device=encoder_hidden_state.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.y_len):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input,
                                                                      decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    # teacher forcing
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, x: torch.Tensor, additional_info: typing.Dict[str, typing.Any]) -> torch.Tensor:
        y: torch.Tensor = additional_info['y']\
            if 'y' in additional_info else None
        batches_seen: int = additional_info['batches_seen']\
            if 'batches_seen' in additional_info else None

        B, x_len, V, x_dim = x.shape
        # x: (B, x_len, V, x_dim)
        x = x.permute(1, 0, 2, 3)
        # x: (x_len, B, V, x_dim)
        x = x.view(x_len, B, -1)
        # x: (x_len, B, V * x_dim)

        if y is not None:
            y = y.permute(1, 0, 2, 3)  # swap N and x_len
            # y: (y_len, B, V, y_dim)
            y = y.view(self.y_len, B, -1)
            # y: (y_len, B, V * y_dim)

        hidden = self._encoder(x)

        y_hat = self._decoder(hidden, y, batches_seen=batches_seen)
        # y_hat: (y_len, B, V * y_dim)
        y_hat = y_hat.view(self.y_len, B, V, -1)
        # y_hat: (y_len, B, V, y_dim)
        y_hat = y_hat.permute(1, 0, 2, 3)  # swap x_len and N
        # y_hat: (B, y_len, V, y_dim)
        return y_hat
