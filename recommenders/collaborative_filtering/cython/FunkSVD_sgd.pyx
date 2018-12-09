cimport cython
cimport numpy as np
import numpy as np
import scipy.sparse as sps

import time
import sys


@cython.boundscheck(False)
def FunkSVD_sgd(R, int num_factors=50, double lrate=0.01,
    double reg=0.015, int n_iterations=10, init_mean=0.0, init_std=0.1, double lrate_decay=1.0, rnd_seed=42):
    if not isinstance(R, sps.csr_matrix):
        raise ValueError('R must be an instance of scipy.sparse.csr_matrix')

    # use Cython MemoryViews for fast access to the sparse structure of R
    cdef int [:] col_indices = R.indices, indptr = R.indptr
    cdef double [:] data = np.array(R.data, dtype=np.float)
    cdef int n_users = R.shape[0], n_items = R.shape[1]
    cdef int nnz = len(R.data)


    # in csr format, indices correspond to column indices
    # let's build the vector of row_indices
    cdef np.ndarray[np.int64_t, ndim=1] row_nnz = np.diff(indptr).astype(np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] row_indices = np.repeat(np.arange(n_users), row_nnz).astype(np.int64)

    # set the seed of the random number generator
    np.random.seed(rnd_seed)

    # randomly initialize the user and item latent factors
    cdef double[:,:] U = np.random.normal(init_mean, init_std, (n_users, num_factors)).astype(np.float)
    cdef double[:,:] V = np.random.normal(init_mean, init_std, (n_items, num_factors)).astype(np.float)

    # build random index to iterate over the non-zero elements in R
    cdef np.ndarray[np.int64_t, ndim=1] shuffled_idx = np.random.permutation(nnz).astype(np.int64)

    # here we define some auxiliary variables
    cdef int i, j, f, idx, currentIteration, numSample
    cdef double rij, rij_pred, err, loss
    #cdef double[:] U_i = np.zeros(num_factors, dtype=np.float)
    #cdef double[:] V_j = np.zeros(num_factors, dtype=np.float)

    start_time_epoch = time.time()
    start_time_batch = time.time()

    #
    # Stochastic Gradient Descent starts here
    #
    for currentIteration in range(n_iterations):     # for each iteration
        loss = 0.0

        for numSample in range(nnz):    # iterate over non-zero values in R only
            idx = shuffled_idx[numSample]
            idx = numSample
            rij = data[idx]

            # get the row and col indices of x_ij
            i = row_indices[idx]
            j = col_indices[idx]

            rij_pred = 0

            # compute the predicted value of R
            for f in range(num_factors):
                #U_i[f] = U[i,f]
                #V_j[f] = V[j,f]
                rij_pred += U[i,f]*V[j,f]


            # compute the prediction error
            err = rij - rij_pred

            # update the loss
            loss += err**2

            # adjust the latent factors
            for f in range(num_factors):
                U[i, f] += lrate * (err * V[j,f] - reg * U[i,f])
                V[j, f] += lrate * (err * U[i,f] - reg * V[j,f])

        loss /= nnz

        # update the learning rate
        lrate *= lrate_decay

        print("Iteration {} of {} completed in {:.2f} minutes. Loss is {:.4f}. Sample per second: {:.0f}".format(
                    currentIteration, n_iterations,
                    (time.time() - start_time_batch)/60,
                    loss,
                    float(nnz) / (time.time() - start_time_batch)))

        sys.stdout.flush()
        sys.stderr.flush()

        start_time_batch = time.time()

    return U, V