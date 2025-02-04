import abc
import typing

import numpy
import torch

from .. import format, matrix


class ModelBase(torch.nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, ctx: dict, dataset: format.TSFMovingWindow, device: torch.device):
        super().__init__()

        self._ctx = ctx
        self._arg = ctx['modelConfig']
        for k, v in self._arg.items():
            setattr(self, k, v)

        self._dataset = dataset
        self._device = device

        self._n_vertex = self._dataset.A.shape[0]

        # remove unnecessary columns
        self._dataset.leave_only_this_columns(ctx['targetSensorAttributes'])

        # create laplacian matrices (=supports)
        if 'adjacencyMatrixLaplacianMatrix' in ctx.keys():
            if ctx['adjacencyMatrixLaplacianMatrix'] is not None:
                if ctx['adjacencyMatrixLaplacianMatrix'] != '':
                    self._supports = self.make_support(
                        self._dataset.A[..., 0],
                        ctx['adjacencyMatrixLaplacianMatrix'],
                        float(ctx['adjacencyMatrixThresholdValue']))

                    self._supports = [torch.from_numpy(A).to(
                        device) for A in self._supports]

    def make_support(self, A: numpy.ndarray, support_type: str, threshold: float) -> typing.List[numpy.ndarray]:
        # check if A is identity matrix (fix NaN bug)
        if numpy.allclose(A, numpy.eye(A.shape[0])):
            return [numpy.eye(A.shape[0]).astype(numpy.float32)]

        if support_type == "identity":
            I = numpy.identity(A.shape[0], dtype=A.dtype)
            return [I]

        A = matrix.thresholded_gaussian_kernel(A, threshold=threshold)

        if support_type == "raw_thresholded_gaussian":
            return [A]

        elif support_type == "norm_lap_sym":
            L = matrix.undirected_adjacency_matrix(A)
            L = matrix.normalized_laplacian_matrix(L)
            return [L]

        elif support_type == "scaled_lap_sym":
            L = matrix.undirected_adjacency_matrix(A)
            L = matrix.scaled_laplacian_matrix(L)
            return [L]

        elif support_type == "rand_mat_asym":
            L = matrix.stochastic_matrix(A, left=True)
            return [L.T]

        elif support_type == "dual_rand_mat_asym":
            L = matrix.stochastic_matrix(A, left=True)
            return [L.T, L]

        elif support_type == "cheb_poly":
            L = matrix.undirected_adjacency_matrix(A)
            if hasattr(self, 'K') and (self.K > 0):
                return matrix.cheb_polynomial(L, self.K)
            else:
                return matrix.cheb_polynomial(L)

        else:
            raise ValueError("support type not defined")

    """
    @abc.abstractmethod
    def forward(self, input: torch.Tensor, label: torch.Tensor = None, batches_seen: int = None, x_time: torch.Tensor = None, y_time: torch.Tensor = None) -> torch.Tensor:
        ...
"""

    '''
    y
    batches_seen
    x_time
    y_time
    '''


    @abc.abstractmethod
    def forward(self, input: torch.Tensor, additional_info: typing.Dict[str, typing.Any] = None) -> torch.Tensor:
        ...