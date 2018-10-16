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



def cos_similarity(sp_icm, sp_rat_m, knn):

    ''' create the prediction matrix using cos_similarity

            param_name | type         | description

    in:     sp_icm     | (csr_matrix) | sparse item-content matrix in CSR format
    in:     sp_rat_m   | (csr_matrix) | sparse rating matrix (in our case URM)
    in:     knn        | (int)        | knn to use for calculate the new similarity matrix
    -----------------------------------------------------
    out:    sp_pred_m  | (csr_matrix) | prediction matrix

    '''

    sp_icm_t = sp_icm.transpose()
    sp_sim_matrix = sp_icm * sp_icm_t

    lil_sim_matrix = sp_sim_matrix.tolil()
    # set the diag of lil matrix to 0
    lil_sim_matrix.setdiag(0)
    sp_sim_matrix = lil_sim_matrix.tocsr()

    sp_sim_matrix_knn = PARROMETHOD(knn)

    sp_pred_m = sp_rat_m * sp_sim_matrix_knn

    return sp_pred_m


def sp_matrix_l2_norm_rows(sp_matrix):

    ''' create a column vector, whithin each cell there is the l2-norm of the corresponding row of the input matrix

            param_name | type              | description

    in:     sp_matrix  | (csr_matrix)[N*M] | sparse matrix of dimension N*M
    -----------------------------------------------------
    out:    l2_vector  | (np_array)[N*1]   | l2-norm vector of dimension N*1

    '''

    l2_vector = np.empty(shape=(sp_matrix.shape[0], 1))
    for i in range(sp_matrix.shape[0]):
        r = sp_matrix.getrow(i)
        _, col_ind = r.nonzero()
        temp = np.empty(shape=(1, col_ind.size))
        for j in col_ind:
            count = 0
            temp[0, count] = r[0, j]
            count += 1
        l2_vector[i] = np.linalg.norm(temp)
    return l2_vector


