from main import Data
from main import M
from cos_similarity import CosineSimilarity
from scipy.sparse import load_npz
from helpers.export import Export
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

#===============================================

sp_urm = load_npz('../dataset/saved_matrices/sp_urm.npz')

U, sigma, Vt = svds(sp_urm, k=50)
sigma = np.diag(sigma)
R = np.dot(np.dot(U, sigma), Vt)
R = R.astype(np.float16)
r_i, c_i = sp_urm.nonzero()
for i in range(len(r_i)):
    k = r_i[i]
    l = c_i[i]
    R[k, l] = -10000
R = np.array(R)
#np.save('../dataset/saved_matrices/R_factorization', R)

#R = csr_matrix(np.load('../dataset/saved_matrices/R_factorization.npy'))
d = Data()
arr_tgt_playlists = d.target_playlists_df.values

n_res_matrix = np.zeros((1, 11))
res = np.ndarray(shape=(1, 11))
n_res = np.array(res)
R = csr_matrix(R)
for i in arr_tgt_playlists:
    r = R.getrow(i)
    for j in range(1, 11):
        c = r.argmax()
        n_res[0, j] = c
        r[0, c] = -10000
    n_res[0, 0] = i
    n_res_matrix = np.concatenate((n_res_matrix, n_res))
n_res_matrix = n_res_matrix[1:, :]

#np.save('../dataset/saved_matrices/res_matrix_fact', n_res_matrix)
Export.export(n_res_matrix, path='submissions/')