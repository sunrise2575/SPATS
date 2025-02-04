import torch
import torch.distributions as tdist

# epsilon for floating point error
# if we don't use epsilon, it will cause NaN error!
EPSILON = torch.finfo(torch.float32).eps


def _masking(y_true: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
    mask = ~((torch.abs(y_true) < EPSILON) | (y_true != y_true)
             | (y_true == torch.inf) | (y_true == -torch.inf))
    # mask = ~((y_true != y_true) | (y_true == torch.inf) | (y_true == -torch.inf)) # has error

    mask = mask.float()
    mask /= mask.mean()

    loss *= mask
    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

    return loss


class MaskedRegressionLossCollection(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, method: str, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # y_pred: (B, y_len, V, y_dim)
        # y_true: (B, y_len, V, y_dim)
        # self.signal_mean: (y_dim,)
        # self.signal_std: (y_dim,)

        assert method in ['MAE', 'MSE', 'RMSE', 'MAPE',
                          'MAAPE', 'MLE'], f'Invalid loss metric: {method}'

        if method == 'MAE':
            loss = torch.abs(torch.sub(y_pred, y_true))
        elif (method == 'MSE') or (method == 'RMSE'):
            loss = torch.pow(torch.sub(y_pred, y_true), 2)
        elif (method == 'MAPE') or (method == 'MAAPE'):
            loss = torch.abs((y_pred - y_true) /
                             torch.clamp(y_true, min=EPSILON))
            if method == 'MAAPE':
                loss = torch.rad2deg(torch.arctan(loss)) / 90.0
        elif method == 'MLE':
            loss = -tdist.Normal(y_pred,
                                 torch.ones_like(y_pred)).log_prob(y_true)

        loss = _masking(y_true, loss)

        if method == 'RMSE':
            return loss.mean().sqrt()
        elif method == 'MLE':
            return loss.sum()
        else:
            return loss.mean()
