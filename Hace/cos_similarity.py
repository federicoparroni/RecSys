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

        m = M()
        sp_sim_matrix_knn = m.create_Sknn(sp_sim_matrix, k=knn)

        sp_pred_m = sp_rat_m * sp_sim_matrix_knn

        return sp_pred_m


