import torch

from .__base import ModelBase


class Model(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.input_dim = self._n_vertex * self.x_dim
        self.output_dim = self._n_vertex * self.y_dim
        self.hidden_dim = self._n_vertex * 1

        self.encoder = torch.nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            # dropout=0.2,
            batch_first=True,
            bidirectional=False,
        )

        self.decoder_cell = torch.nn.LSTMCell(
            input_size=self.output_dim,
            hidden_size=self.hidden_dim)

        self.dropout = torch.nn.Dropout(p=0.2)

        self.linear = torch.nn.Linear(
            in_features=self.hidden_dim,
            out_features=self.output_dim)

        self.activation = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor, *_) -> torch.Tensor:
        B, x_len, V, x_dim = x.shape

        x = x.view(B, x_len, self.input_dim)
        # x = [B, x_len, self.input_dim]

        # Encoder
        _, encoded = self.encoder(input=x, hx=None)
        # encoded is a tuple of (output, (h_n, c_n))
        # encoded = [1, B, self.hidden_dim]
        h = encoded[0].squeeze(0)
        c = encoded[1].squeeze(0)
        # encoded = [B, self.hidden_dim]

        # Decoder
        x = torch.zeros(B, self.output_dim).to(x.device)
        outputs = []

        for t in range(self.y_len):
            h, c = self.decoder_cell(input=x, hx=(h, c))
            # cur_output = [B, self.hidden_dim]

            y = self.activation(h)
            # h = [B, self.hidden_dim]
            y = self.dropout(y)
            # h = [B, self.hidden_dim]
            y = self.linear(y)
            # h = [B, self.output_dim]

            x = y

            outputs.append(y.view(B, V, self.y_dim))

        result = torch.stack(outputs, dim=1)
        # result = [B, self.y_len, V, self.y_dim]

        return result
