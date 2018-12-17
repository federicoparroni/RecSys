import utils.check_matrix_format as cm
import scipy.sparse as sps
import numpy as np
import time

def apply_top_k(matrix, k):
    start = time.time()
    matrix = cm.check_matrix(matrix, format='csr')
    # initializing the new row to substitue to the preciding one
    filtered_matrix = np.empty(shape=(matrix.shape[0], matrix.shape[1]))
    for i in range(matrix.shape[0]):

        row = matrix.getrow(i)
        row = row.todense()
        relevant_items_row_indices = (-row).argpartition(k)[0, 0:k]
        for c_index in relevant_items_row_indices[0]:
            filtered_matrix[i,c_index] = row[0,c_index]
    #convert the matrix to a sparse format
    sp_filtered_matrix = sps.csr_matrix(filtered_matrix)
    print('topK applied in {} s'.format(time.time()-start))
    return sp_filtered_matrix


if __name__ == '__main__':
    sim_matrix = sps.load_npz('raw_data/saved_sim_matrix_evaluation/CFitem.npz')
    apply_top_k(sim_matrix, 100)

        