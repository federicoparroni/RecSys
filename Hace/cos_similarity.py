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

from main import M

#===============================================


class CosineSimilarity:

    @staticmethod
    def predict(sp_icm, sp_rat_m, knn):

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
        l2_norm_vector = CosineSimilarity.sp_matrix_l2_norm_rows(sp_sim_matrix)
        m = M()
        sp_sim_matrix_knn = m.create_Sknn(sp_sim_matrix, k=knn)

        sp_pred_m = sp_rat_m * sp_sim_matrix_knn

        return sp_pred_m


    @staticmethod
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


    @staticmethod
    def normalize_sp_sim_matrix(sp_sim_matrix):

        ''' normalize the element of the similarity matrix with the l2-norm

                param_name    | type              | description

        in:     sp_sim_matrix | (csr_matrix)[N*M] | sparse matrix of dimension N*M (have senso only for the sim matrix)
        -----------------------------------------------------
        out:    _             | _                 | the matrix passed as parameter will be normalized by l2_norm

        '''

        l2_vect = CosineSimilarity.sp_matrix_l2_norm_rows(sp_sim_matrix)
        r_i, c_i = sp_sim_matrix.nonzero()
        for i in range(len(r_i)):
            k = r_i[i]
            l = c_i[i]
            sp_sim_matrix[k, l] = (sp_sim_matrix[k, l]/(l2_vect[k]*l2_vect[l]))


"""
#test normalize_sp_sim_matrix
sp_icm = load_npz('../dataset/saved_matrices/sp_icm.npz')

sp_icm_t = sp_icm.transpose()
sp_sim_matrix = sp_icm * sp_icm_t
CosineSimilarity.normalize_sp_sim_matrix(sp_sim_matrix)
a= 5
"""
