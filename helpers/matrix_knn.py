import numpy as np

def getKnn(A, k = 10):
    """
    given a sparse csr matrix A, gets back the same matrix but keeping for each row just the k greatest values and
    setting the rest to 0
    @params:
        A           - (csr_matrix) matrix to which apply the knn by rows
        k           - (int) k to which
    @returns:
        -           - (csr_matrix) knn matrix to which knn has been applied by rows
    """

    for i in range(A.shape[0]):
        # Get the row slice, not a copy, only the non zero elements
        row_array = A.data[A.indptr[i]: A.indptr[i+1]]
        if row_array.shape[0] <= k:
            # Not more than k elements
            continue

        # only take the six last k elements in the sorted indeces
        row_array[np.argsort(row_array)[:-k]] = 0

    A.eliminate_zeros()
    return A
