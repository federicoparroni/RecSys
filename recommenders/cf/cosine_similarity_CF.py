import numpy as np
#from matrix import M
from scipy.sparse import load_npz
from scipy import sparse
import math
# ===============================================

class CosineSimilarityCF:

    @staticmethod
    def predict(sp_urm, knn=-1, shrink_term=0):

        ''' create the prediction matrix using cos_similarity

                param_name | type         | description

        in:     sp_icm     | (csr_matrix) | sparse item-content matrix in CSR format
        in:     sp_rat_m   | (csr_matrix) | sparse rating matrix (in our case URM)
        in:     knn        | (int)        | knn to use for calculate the new similarity matrix
        in:     shrink_term  | (int)        | integer used for normalization
        -----------------------------------------------------
        out:    sp_pred_m  | (csr_matrix) | prediction matrix
        '''

        sp_sim_matrix = sp_urm * sp_urm.transpose()
        print('sim matrix computed')

        sp_sim_matrix.setdiag(0)

        print('diag set to 0')

        CosineSimilarityCF.normalize_sp_sim_matrix(sp_sim_matrix, shrink_term)
        print('matrix normalized')

        m = M()
        sp_sim_matrix_knn = m.create_Sknn(sp_sim_matrix, k=knn)
        print('knn done')

        sp_pred_m = sp_sim_matrix_knn * sp_urm
        print('pred mat done')

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

            """
            _, col_ind = r.nonzero()
            temp = np.empty(shape=(1, col_ind.size))
            for j in col_ind:
                count = 0
                temp[0, count] = r[0, j]
                count += 1
            l2_vector[i] = np.linalg.norm(temp)
            """
        return l2_vector


    @staticmethod
    def normalize_sp_sim_matrix(sp_sim_matrix, shrink_term=0):

        ''' normalize the element of the similarity matrix with the l2-norm

                param_name    | type              | description

        in:     sp_sim_matrix | (csr_matrix)[N*M] | sparse matrix of dimension N*M (have senso only for the sim matrix)
        -----------------------------------------------------
        out:    _             | _                 | the matrix passed as parameter will be normalized by l2_norm

        '''

        l2_vect = CosineSimilarityCF.sp_matrix_l2_norm_rows(sp_sim_matrix)
        print('done l2_vect')
        #divide each row and column for the correspoding element for the normalization task
        for i in range(len(l2_vect)):
            new_row = sp_sim_matrix.getrow(i).multiply(1/l2_vect[i]).todense()
            new_row = np.array(new_row)[0]
            CosineSimilarityCF.set_row_csr(sp_sim_matrix, i, new_row)
        sp_sim_matrix = sp_sim_matrix.transpose()
        for i in range(len(l2_vect)):
            new_row = sp_sim_matrix.getrow(i).multiply(1/l2_vect[i]).todense()
            new_row = np.array(new_row)[0]
            CosineSimilarityCF.set_row_csr(sp_sim_matrix, i, new_row)
        sp_sim_matrix = sp_sim_matrix.transpose()
        #use the shrink term in the normalization
        if(shrink_term != 0):
            for i in range(len(l2_vect)):
                new_row = sp_sim_matrix.getrow(i).multiply(1/shrink_term).todense()
                new_row = np.array(new_row)[0]
                CosineSimilarityCF.set_row_csr(sp_sim_matrix, i, new_row)


        """
        #divide each column for the l2norm of it
        
        r_i, c_i = sp_sim_matrix.nonzero()
        for i in range(len(r_i)):
            k = r_i[i]
            l = c_i[i]
            sp_sim_matrix[k, l] = (sp_sim_matrix[k, l]/(l2_vect[k]*l2_vect[l]+shrink_term))
        """

    @staticmethod
    def set_row_csr(A, row_idx, new_row):
        '''
        Replace a row in a CSR sparse matrix A.

        Parameters
        ----------
        A: csr_matrix
            Matrix to change
        row_idx: int
            index of the row to be changed
        new_row: np.array
            list of new values for the row of A

        Returns
        -------
        None (the matrix A is changed in place)

        Prerequisites
        -------------
        The row index shall be smaller than the number of rows in A
        The number of elements in new row must be equal to the number of columns in matrix A
        '''
        assert sparse.isspmatrix_csr(A), 'A shall be a csr_matrix'
        assert row_idx < A.shape[0], \
            'The row index ({0}) shall be smaller than the number of rows in A ({1})' \
                .format(row_idx, A.shape[0])
        try:
            N_elements_new_row = len(new_row)
        except TypeError:
            msg = 'Argument new_row shall be a list or numpy array, is now a {0}' \
                .format(type(new_row))
            raise AssertionError(msg)
        N_cols = A.shape[1]
        assert N_cols == N_elements_new_row, \
            'The number of elements in new row ({0}) must be equal to ' \
            'the number of columns in matrix A ({1})' \
                .format(N_elements_new_row, N_cols)

        idx_start_row = A.indptr[row_idx]
        idx_end_row = A.indptr[row_idx + 1]
        additional_nnz = N_cols - (idx_end_row - idx_start_row)

        A.data = np.r_[A.data[:idx_start_row], new_row, A.data[idx_end_row:]]
        A.indices = np.r_[A.indices[:idx_start_row], np.arange(N_cols), A.indices[idx_end_row:]]
        A.indptr = np.r_[A.indptr[:row_idx + 1], A.indptr[(row_idx + 1):] + additional_nnz]



#test normalize_sp_sim_matrix
# sp_urm = load_npz('../../raw_data/matrices/urm.npz')
# print('loaded matrix')
#
# sp_urm_t = sp_urm.transpose()
# print('matrix transposed')
#
# sp_sim_matrix = sp_urm * sp_urm_t
# print('sim matrix computed')
#
# CosineSimilarityCF.normalize_sp_sim_matrix(sp_sim_matrix, 10)
# a = 5

