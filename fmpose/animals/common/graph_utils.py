from __future__ import absolute_import

import torch
import numpy as np
import scipy.sparse as sp

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)) # 1 压缩列。将每一行元素相加
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0. # 将+/-inf 置为0 
    r_mat_inv = sp.diags(r_inv) # 构造对角阵
    mx = r_mat_inv.dot(mx) # 
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adj_mx_from_edges(num_pts, edges, sparse=True):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1] #  j：[0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 7, 11, 8, 7
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32) # 17,17
    # build symmetric adjacency matrix  https://github.com/yao8839836/text_gcn/issues/17 
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))
    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
    return adj_mx 


def adj_mx_from_skeleton(skeleton):
    num_joints = skeleton.num_joints() # 16|17
    # edge [16,2]
    edges = list(filter(lambda x: x[1] >= 0, zip(list(range(0, num_joints)), skeleton.parents()))) # 15  # [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 7, 11, 12, 7, 14, 15]
    return adj_mx_from_edges(num_joints, edges, sparse=False)


def print_matrix(mat):
    for i in range(len(mat)):
        print(mat[i])

if __name__=="__main__":
    num_joints = 17
    parents =  [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 7, 11, 12, 7, 14, 15]
    edges = list(filter(lambda x: x[1] >= 0, zip(list(range(0, num_joints)), parents)))
    A = adj_mx_from_edges(num_joints, edges, sparse=False)
    print_matrix(A)
