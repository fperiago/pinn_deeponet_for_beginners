import numpy as np


def compute_eig(kernel, num, Nx, eigenfunction=True):
    """
    Computes the eigenvalues and ortonormalized eigenfunctions of a kernel on [0, 1].
    Returns a tuple eigval, eigvec, where eigval contains the num largest eigenvalues, and
    eigvec is a matrix of dimension Nx times num. Column j corresponds to
    eigenfuncion associated to eigenvalue eigval[j], evaluated at np.linspace(0, 1, num=Nx).
    """
    # We consider a discretization grid of width 0.01 on [0, 1]
    num_discretization = np.max([100, Nx]) + 1
    if num_discretization < num:
        raise ValueError(
            """The grid in the kernel discretization to compute the eigenfunctions is not sufficient.
            There are more than one hundred terms required in the KL expansion"""
        )
    h = 1 / (num_discretization - 1) # discretization step 
    # first row of the kernel times h (trapezoidal integration) Kernel(0, x)*h, x=np.linspace(0, 1, num=num_discretization)
    c = kernel(np.linspace(0, 1, num=num_discretization)[:, None])[0] * h 
    A = np.empty((num_discretization, num_discretization))
    for i in range(num_discretization): # kernel's discretization
        A[i, i:] = c[: num_discretization - i]
        A[i, i::-1] = c[: i + 1] # shift to the right one position
    A[:, 0] *= 0.5 # trapezoidal correction
    A[:, -1] *= 0.5 # trapezoidal correction

    if not eigenfunction:
        return np.flipud(np.sort(np.real(np.linalg.eigvals(A))))[:num]

    eigval, eigvec = np.linalg.eig(A) # computation of eigenvalues and eigenvectors
    eigval, eigvec = np.real(eigval), np.real(eigvec)
    idx = np.flipud(np.argsort(eigval))[:num]
    eigval, eigvec = eigval[idx], eigvec[:, idx]
    integrals = np.trapz(eigvec ** 2, dx=h, axis=0) ** 0.5
    eigvec /= integrals.reshape(1, -1) # normalization of eigenvectors
    indexes = (np.round(np.linspace(0, 1, num=Nx), 2) * 100).astype(int)
    return eigval, eigvec[indexes, :]
