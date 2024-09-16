''' The construction of projection method is modified from SciPy, see more information in SciPy '''

from scipy.sparse import (bmat, csc_matrix, eye, issparse)
from scipy.sparse.linalg import LinearOperator
import scipy.linalg
import scipy.sparse.linalg
try:
    from sksparse.cholmod import cholesky_AAt
    sksparse_available = True
except ImportError:
    import warnings
    sksparse_available = False
import numpy as np
from warnings import warn



def orthogonality(A, g):
    norm_g = np.linalg.norm(g)
    if issparse(A):
        norm_A = scipy.sparse.linalg.norm(A, ord='fro')
    else:
        norm_A = np.linalg.norm(A, ord='fro')
    if norm_g == 0 or norm_A == 0:
        return 0

    norm_A_g = np.linalg.norm(A.dot(g))
    orth = norm_A_g / (norm_A*norm_g)
    return orth


def normal_equation_projections(A, m, n, orth_tol, max_refin, tol):
    factor = cholesky_AAt(A)
    def null_space(x):
        v = factor(A.dot(x))
        z = x - A.T.dot(v)
        k = 0
        while orthogonality(A, z) > orth_tol:
            if k >= max_refin:
                break
            v = factor(A.dot(z))
            z = z - A.T.dot(v)
            k += 1

        return z

    def least_squares(x):
        return factor(A.dot(x))

    def row_space(x):
        return A.T.dot(factor(x))

    return null_space, least_squares, row_space


def augmented_system_projections(A, m, n, orth_tol, max_refin, tol):
    K = csc_matrix(bmat([[eye(n), A.T], [A, None]]))
    try:
        solve = scipy.sparse.linalg.factorized(K)
    except RuntimeError:
        warn("Singular Jacobian matrix. Using dense SVD decomposition to "
             "perform the factorizations.",
             stacklevel=3)
        return svd_factorization_projections(A.toarray(),
                                             m, n, orth_tol,
                                             max_refin, tol)
    def null_space(x):
        v = np.hstack([x, np.zeros(m)])
        lu_sol = solve(v)
        z = lu_sol[:n]
        k = 0
        while orthogonality(A, z) > orth_tol:
            if k >= max_refin:
                break
            new_v = v - K.dot(lu_sol)
            lu_update = solve(new_v)
            lu_sol += lu_update
            z = lu_sol[:n]
            k += 1
        return z


    def least_squares(x):
        v = np.hstack([x, np.zeros(m)])
        lu_sol = solve(v)
        return lu_sol[n:m+n]

    def row_space(x):
        v = np.hstack([np.zeros(n), x])
        lu_sol = solve(v)
        return lu_sol[:n]

    return null_space, least_squares, row_space


def qr_factorization_projections(A, m, n, orth_tol, max_refin, tol):
    Q, R, P = scipy.linalg.qr(A.T, pivoting=True, mode='economic')

    if np.linalg.norm(R[-1, :], np.inf) < tol:
        warn('Singular Jacobian matrix. Using SVD decomposition to ' +
             'perform the factorizations.',
             stacklevel=3)
        return svd_factorization_projections(A, m, n,
                                             orth_tol,
                                             max_refin,
                                             tol)
    def null_space(x):
        aux1 = Q.T.dot(x)
        aux2 = scipy.linalg.solve_triangular(R, aux1, lower=False)
        v = np.zeros(m)
        v[P] = aux2
        z = x - A.T.dot(v)
        k = 0
        while orthogonality(A, z) > orth_tol:
            if k >= max_refin:
                break
            aux1 = Q.T.dot(z)
            aux2 = scipy.linalg.solve_triangular(R, aux1, lower=False)
            v[P] = aux2
            z = z - A.T.dot(v)
            k += 1

        return z

    def least_squares(x):
        aux1 = Q.T.dot(x)
        aux2 = scipy.linalg.solve_triangular(R, aux1, lower=False)
        z = np.zeros(m)
        z[P] = aux2
        return z

    def row_space(x):
        aux1 = x[P]
        aux2 = scipy.linalg.solve_triangular(R, aux1,
                                             lower=False,
                                             trans='T')
        z = Q.dot(aux2)
        return z

    return null_space, least_squares, row_space


def svd_factorization_projections(A, m, n, orth_tol, max_refin, tol):
    U, s, Vt = scipy.linalg.svd(A, full_matrices=False)
    U = U[:, s > tol]
    Vt = Vt[s > tol, :]
    s = s[s > tol]
    def null_space(x):
        aux1 = Vt.dot(x)
        aux2 = 1/s*aux1
        v = U.dot(aux2)
        z = x - A.T.dot(v)
        k = 0
        while orthogonality(A, z) > orth_tol:
            if k >= max_refin:
                break
            aux1 = Vt.dot(z)
            aux2 = 1/s*aux1
            v = U.dot(aux2)
            z = z - A.T.dot(v)
            k += 1

        return z

    def least_squares(x):
        aux1 = Vt.dot(x)
        aux2 = 1/s*aux1
        z = U.dot(aux2)
        return z

    def row_space(x):
        aux1 = U.T.dot(x)
        aux2 = 1/s*aux1
        z = Vt.T.dot(aux2)
        return z

    return null_space, least_squares, row_space


def projections(A, method=None, orth_tol=1e-12, max_refin=3, tol=1e-15):
    m, n = np.shape(A)
    if m*n == 0:
        A = csc_matrix(A)
    if issparse(A):
        if method is None:
            method = "AugmentedSystem"
        if method not in ("NormalEquation", "AugmentedSystem"):
            raise ValueError("Method not allowed for sparse matrix.")
        if method == "NormalEquation" and not sksparse_available:
            warnings.warn("Only accepts 'NormalEquation' option when "
                          "scikit-sparse is available. Using "
                          "'AugmentedSystem' option instead.",
                          ImportWarning, stacklevel=3)
            method = 'AugmentedSystem'
    else:
        if method is None:
            method = "QRFactorization"
        if method not in ("QRFactorization", "SVDFactorization"):
            raise ValueError("Method not allowed for dense array.")

    if method == 'NormalEquation':
        null_space, least_squares, row_space \
            = normal_equation_projections(A, m, n, orth_tol, max_refin, tol)
    elif method == 'AugmentedSystem':
        null_space, least_squares, row_space \
            = augmented_system_projections(A, m, n, orth_tol, max_refin, tol)
    elif method == "QRFactorization":
        null_space, least_squares, row_space \
            = qr_factorization_projections(A, m, n, orth_tol, max_refin, tol)
    elif method == "SVDFactorization":
        null_space, least_squares, row_space \
            = svd_factorization_projections(A, m, n, orth_tol, max_refin, tol)

    Z = LinearOperator((n, n), null_space)
    LS = LinearOperator((m, n), least_squares)
    Y = LinearOperator((n, m), row_space)
    return Z, LS, Y


