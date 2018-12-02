cimport cython
cimport numpy as np
import numpy as np
import scipy.sparse as sps

import time
import sys

@cython.boundscheck(False)
def factor_matrix(R, K, alpha, steps, beta, error_limit):

    # use Cython MemoryViews for fast access to the sparse structure of R
    cdef int N = R.shape[0], M = R.shape[1]

    # randomly initialize the item and users latent factors
    cdef np.ndarray[np.float32_t, ndim=2] P = np.random.rand(N, K).astype(np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] Q = np.random.rand(M, K).astype(np.float32)

    # here we define some auxiliary variables
    cdef int i, j, k, l, it
    cdef err


    #transpose Q
    Q = Q.T

    #
    # Stochastic Gradient Descent starts here
    #

    # iterate through max # of steps
    for it in range(steps):
        print('performing the {} iteration'.format(it))

        # iterate each cell in r
        for i in range(N):
            for j in range(M):
                if R[i, j] > 0:
                    print(i)

                    # get the eij (error) side of the equation
                    eij = R[i, j] - np.dot(P[i, :], Q[:, j])

                    for k in range(K):
                        # (*update_rule) update pik_hat
                        P[i, k] = P[i, k] + alpha * (2 * eij * Q[k, j] - beta * P[i, k])

                        # (*update_rule) update qkj_hat
                        Q[k, j] = Q[k, j] + alpha * (2 * eij * P[i, k] - beta * Q[k, j])

        # Measure error
        print('calculate the loss')

        err = 0
        for k in range(N):
            for l in range(M):
                if R[i, j] > 0:

                    # loss function error sum( (y-y_hat)^2 )
                    err = err + pow(R[i, j]-np.dot(P[i, :], Q[:, j]), 2)

                    # add regularization
                    for k in range(K):

                        # error + ||P||^2 + ||Q||^2
                        err = err + (beta/2) * (pow(P[i, k], 2) + pow(Q[k, j], 2))
        print('loss at iteration {} is: {}'.format(it, err))

        # Terminate when we converge
        if err < error_limit:
            break

    # track Q, P (learned params)
    # Q = Products x feature strength
    # P = Users x feature strength
    return Q.T, P
