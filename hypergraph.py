import numpy as np
import scipy
from scipy.sparse.linalg import cg
from sklearn.neighbors import NearestNeighbors


def normalize(A):
    '''Row-normalize sparse matrix'''
    rowsum = np.array(A.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = scipy.sparse.diags(r_inv)
    A = r_mat_inv.dot(A)
    return A


def build_hypergraph(X, n_neighbors, metric='cosine'):
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(X)
    H = knn.kneighbors_graph(X)
    return H


def label_propagation(adj, y_train, idx_train, idx_test, pred_train=None, pred_test=None):
    '''
    Attributes
    ----------
    adj: scipy.sparse
        adjacency matrix
    '''
    adj = normalize(adj)
    L = scipy.sparse.eye(adj.shape[0]) - adj
    z = np.zeros((len(idx_test), 1))
    if pred_train is None:
        res_train = y_train
    else:
        res_train = y_train - pred_train
    A = L[idx_test, :][:, idx_test]
    b = -L[idx_test, :][:, idx_train] @ res_train
    res_test = cg(A, b)[0].reshape(-1, 1)
    if pred_test is None:
        z = res_test
    else:
        z = pred_test + res_test
    return z


def hyper_propagation(X, y_train, idx_train, idx_test, n_neighbors, pred_train=None, pred_test=None):
    H = build_hypergraph(X, n_neighbors)
    z = label_propagation(H, y_train, idx_train, idx_test, pred_train, pred_test)
    return z