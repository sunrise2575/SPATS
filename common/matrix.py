import numpy
import numpy.linalg
import typing


def thresholded_gaussian_kernel(A: numpy.ndarray, threshold=0.1) -> numpy.ndarray:
    # function from DCRNN
    assert threshold >= 0 and threshold <= 1, "threshold must be in [0, 1]"
    distance = A[~numpy.isinf(A)].flatten()
    std = distance.std()
    result = numpy.exp(-numpy.square(A / std))
    result[result < threshold] = 0
    return result


def undirected_adjacency_matrix(A: numpy.ndarray) -> numpy.ndarray:
    return numpy.maximum.reduce([A, A.T])

def normalized_adjacency_matrix(A: numpy.ndarray, indegree: bool = False) -> numpy.ndarray:
    axis = 0 if indegree else 1
    diags = A.sum(axis=axis)
    D = numpy.diag(diags)

    D_inv = numpy.linalg.pinv(D) # Moore-Penrose inverse
    D_inv_sqrt = numpy.sqrt(D_inv) # (D+)^(1/2)

    return D_inv_sqrt.dot(A.dot(D_inv_sqrt))


def laplacian_matrix(A: numpy.ndarray, indegree: bool = False) -> numpy.ndarray:
    # https://en.wikipedia.org/wiki/Laplacian_matrix#Laplacian_matrix_2
    axis = 0 if indegree else 1
    diags = A.sum(axis=axis)
    D = numpy.diag(diags)
    L = D - A

    return L


def normalized_laplacian_matrix(A: numpy.ndarray, indegree: bool = False) -> numpy.ndarray:
    # https://en.wikipedia.org/wiki/Laplacian_matrix#Symmetrically_normalized_Laplacian_2
    axis = 0 if indegree else 1
    diags = A.sum(axis=axis)
    D = numpy.diag(diags)
    L = D - A

    D_inv = numpy.linalg.pinv(D) # Moore-Penrose inverse
    D_inv_sqrt = numpy.sqrt(D_inv) # (D+)^(1/2)

    return D_inv_sqrt.dot(L.dot(D_inv_sqrt))


def scaled_laplacian_matrix(A: numpy.ndarray, indegree: bool = False) -> numpy.ndarray:
    # rescaling Laplacian matrix from [0, lambda_max] to [-1, 1]
    # originally shown in Appendix section C of https://arxiv.org/pdf/1707.01926.pdf;
    #     "... represents a rescaling of the graph Laplacian that maps ..."

    axis = 0 if indegree else 1
    L = normalized_laplacian_matrix(A, axis)
    
    lambda_max = numpy.linalg.eigvalsh(L).max()
    # same as: linalg.eigsh(L, 1, which='LM')
    #     1: number of eigenvalues
    #     LM: Largest (in magnitude) eigenvalues
    #     https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html

    if lambda_max < 0.0001:
        lambda_max = 2 # prevent division by zero

    with numpy.errstate(divide='ignore'):
        return ((2 / lambda_max) * L) - numpy.identity(A.shape[0], dtype=A.dtype)

def stochastic_matrix(A: numpy.ndarray, left: bool = False) -> numpy.ndarray:
    # https://en.wikipedia.org/wiki/Laplacian_matrix#Random_walk_normalized_Laplacian
    axis = 0 if left else 1
    diags = A.sum(axis=axis)
    D = numpy.diag(diags)
    D_inv = numpy.linalg.pinv(D) # Moore-Penrose inverse

    if left:
        return A.dot(D_inv) # left stochastic matrix
    else:
        return D_inv.dot(A) # right stochastic matrix

def random_walk_laplacian_matrix(A: numpy.ndarray, indegree: bool = False) -> numpy.ndarray:
    # https://en.wikipedia.org/wiki/Laplacian_matrix#Random_walk_normalized_Laplacian
    axis = 0 if indegree else 1
    diags = A.sum(axis=axis)
    D = numpy.diag(diags)
    D_inv = numpy.linalg.pinv(D) # Moore-Penrose inverse
    L = D - A

    if indegree:
        return L.dot(D_inv) # right normalized Laplacian
    else:
        return D_inv.dot(L) # left normalized Laplacian


def cheb_polynomial(A: numpy.ndarray, K: int = 3) -> typing.List[numpy.ndarray]:
    # code from ASTGCN
    #compute a list of chebyshev polynomials from T_0 to T_{K-1} 
    #A: adjacency matrix (V, V)
    #L_tilde: scaled Laplacian, numpy.ndarray, shape (V, V)
    #K: the maximum order of chebyshev polynomials

    #cheb_polynomials: list(numpy.ndarray), length: K, from T_0 to T_{K-1}

    L_tilde = scaled_laplacian_matrix(A)

    cheb_polynomials = [numpy.identity(A.shape[0], A.dtype), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials