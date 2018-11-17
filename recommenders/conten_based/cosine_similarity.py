import numpy as np
import math
from recommenders.matrix_knn import getKnn
# ===============================================

class CosineSimilarityCB:

    @staticmethod
    def predict(sp_icm, sp_rat_m, knn=-1, shrink_term=0):

        ''' create the prediction matrix using cos_similarity

                param_name  | type         | description

        in:     sp_icm       | (csr_matrix) | sparse item-content matrix in CSR format
        in:     sp_rat_m     | (csr_matrix) | sparse rating matrix (in our case URM)
        in:     knn          | (int)        | knn to use for calculate the new similarity matrix
        in:     shrink_term  | (int)        | integer used for normalization
        -----------------------------------------------------
        out:    sp_pred_m  | (csr_matrix) | prediction matrix
        '''

        sp_sim_matrix = sp_icm * sp_icm.transpose()

        sp_sim_matrix.setdiag(0)

        CosineSimilarityCB.normalize_sp_sim_matrix(sp_sim_matrix, shrink_term)

        sp_sim_matrix_knn = getKnn(sp_sim_matrix, k=knn)

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
            r_t = r.transpose()
            temp = r*r_t
            l2_vector[i] = math.sqrt(temp[0, 0])
        return l2_vector


    @staticmethod
    def normalize_sp_sim_matrix(sp_sim_matrix, shrink_term=0):

        ''' normalize the element of the similarity matrix with the l2-norm and optionally with a shrink term

                          param_name    | type              | description

                  in:     sp_sim_matrix | (csr_matrix)[N*M] | sparse matrix of dimension N*M (have senso only for the sim matrix)
        (optional)in:     shrink_term   | (INT)             | constant at denominator for normalization purpose
        -----------------------------------------------------
                 out:    _             | _                 | the matrix passed as parameter will be normalized by l2_norm

        '''

        l2_vect = CosineSimilarityCB.sp_matrix_l2_norm_rows(sp_sim_matrix)

        r_i, c_i = sp_sim_matrix.nonzero()
        for i in range(len(r_i)):
            k = r_i[i]
            l = c_i[i]
            sp_sim_matrix[k, l] = (sp_sim_matrix[k, l]/(l2_vect[k]*l2_vect[l]+shrink_term))


# d = Data()
# m = M()
#
# sp_urm = load_npz('../raw_data/matrices/sp_urm_train_MAP.npz')
# sp_icm = load_npz('../raw_data/matrices/icm.npz')
# print('loaded matrices')
#
# sp_pred_mat1 = CosineSimilarityCB.predict(sp_icm, sp_urm, knn=10, shrink_term=30)
# print('computed estimated ratings')
#
# bestn = get_best_n_ratings(sp_pred_mat1, d.target_playlists_df, sp_urm)
# print('got the best n ratings for the target playlists')
#
# Export.export(np.array(bestn), path='../Hace/submissions/', name='content_based')
# print('exported')
