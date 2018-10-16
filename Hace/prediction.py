from main import Data
from main import M
from cos_similarity import CosineSimilarity
from scipy.sparse import load_npz
from helpers.export import Export
import numpy as np

#===============================================

d = Data()
m = M()

sp_urm = load_npz('../dataset/saved_matrices/sp_urm.npz')
sp_icm = load_npz('../dataset/saved_matrices/sp_icm.npz')

sp_pred_mat = CosineSimilarity.predict(sp_icm, sp_urm, knn=50)

arr_tgt_playlists = d.target_playlists_df.values

n_res_matrix = np.zeros((1, 11))
res = np.ndarray(shape=(1, 11))
n_res = np.array(res)

for i in arr_tgt_playlists:
    r = sp_pred_mat.getrow(i)
    for j in range(1, 11):
        c = r.argmax()
        n_res[0, j] = c
        r[0, c] = 0
    n_res[0, 0] = i
    n_res_matrix = np.concatenate((n_res_matrix, n_res))
n_res_matrix = n_res_matrix[1:, :]

Export.export(n_res_matrix, path='submissions/')
