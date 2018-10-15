from main import Data
from main import M
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
from scipy.sparse import load_npz
from scipy.sparse import lil_matrix
from helpers.export import Export
import datetime

import numpy as np

#===============================================

d = Data()
m = M()

"""
#code that have been used for create the URM create a sparse representation of it and save it to a file

urm = m.create_urm(d.playlists_df)
sp_urm = csr_matrix(urm)
save_npz('saved_matrices/sp_urm', sp_urm)


#creating icm removing the duration feature

icm = m.create_icm(d.tracks_df.filter(items=['track_id', 'album_id', 'artist_id']))
sp_icm = csr_matrix(icm)
save_npz('saved_matrices/sp_icm', sp_icm)

"""

sp_urm = load_npz('../dataset/saved_matrices/sp_urm.npz')
sp_icm = load_npz('../dataset/saved_matrices/sp_icm.npz')

sp_icm_t = sp_icm.transpose()

sp_sim_matrix = sp_icm * sp_icm_t

lil_sim_matrix = sp_sim_matrix.tolil()
#set the diag of lil matrix to 0
lil_sim_matrix.setdiag(0)

sp_sim_matrix = lil_sim_matrix.tocsr()

sp_extimation_m = sp_urm*sp_sim_matrix


arr_tgt_playlists = d.target_playlists_df.values


n_res_matrix = np.zeros((1, 11))

res = np.ndarray(shape=(1, 11))
n_res = np.array(res)

for i in arr_tgt_playlists:
    r = sp_extimation_m.getrow(i+1)
    for j in range(1, 11):
        c = r.argmax()
        n_res[0, j] = c
        r[0, c] = 0
    n_res[0, 0] = i
    n_res_matrix = np.concatenate((n_res_matrix, n_res))
n_res_matrix = n_res_matrix[1:, :]

Export.export(n_res_matrix, path='submissions/')

