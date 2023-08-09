import math
import numpy as np
from scipy import linalg as la

def PCA(data, dims_rescaled_data=2):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    mn = np.mean(data, axis=0)
    # mean center the data
    data -= mn
    # calculate the covariance matrix
    C = np.cov(data.T)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    evals, evecs = la.eig(C)
    # sorted them by eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:,:dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    data_rescaled = np.dot(evecs.T, data.T).T
    # reconstruct original data array
    data_original_regen = np.dot(evecs, dim1).T + mn
    return data_rescaled, data_original_regen


def plot_pca(data):
    import matplotlib as mpl
    clr1 =  '#2026B2'
    fig = mpl.figure()
    ax1 = fig.add_subplot(111)
    data_resc, data_orig = PCA(data)
    ax1.plot(data_resc[:,0], data_resc[:,1], '.', mfc=clr1, mec=clr1)
    mpl.show()

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
