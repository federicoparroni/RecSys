cimport cython
cimport numpy as np
import numpy as np
import scipy.sparse as sps

import time
import sys

from libc.math cimport exp, log

@cython.boundscheck(False)
def BPRMF_sgd(R, num_factors=50, lrate=0.01, user_reg=0.015, pos_reg=0.015, neg_reg=0.0015, iters=10,
              sampling_type='user_uniform_item_uniform',sample_with_replacement=True, use_resampling=False, sampling_pop_alpha=1.0,
     init_mean=0.0, init_std=0.1, lrate_decay=1.0, rnd_seed=42,verbose=False):
    if not isinstance(R, sps.csr_matrix):
        raise ValueError('R must be an instance of scipy.sparse.csr_matrix')

    # use Cython MemoryViews for fast access to the sparse structure of R
    cdef int [:] col_indices = R.indices, indptr = R.indptr
    cdef float [:] data = R.data
    cdef int M = R.shape[0], N = R.shape[1]
    cdef int nnz = len(R.data)

    # set the seed of the random number generator
    np.random.seed(rnd_seed)
    # randomly initialize the user and item latent factors
    cdef np.ndarray[np.float32_t, ndim=2] X = np.random.normal(init_mean, init_std, (M, num_factors)).astype(np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] Y = np.random.normal(init_mean, init_std, (N, num_factors)).astype(np.float32)

    # sample the training triples
    cdef np.ndarray[np.int64_t, ndim=2] sample
    if sampling_type == 'user_uniform_item_uniform':
        sample = user_uniform_item_uniform_sampling(R, nnz, replace=sample_with_replacement, seed=rnd_seed, verbose=verbose)
    elif sampling_type == 'user_uniform_item_pop':
        sample = user_uniform_item_pop_sampling(R, nnz, alpha=sampling_pop_alpha, seed=rnd_seed, verbose=verbose)
    else:
        raise RuntimeError('Unknown sampling procedure "{}"'.format(sampling_type))

    # here we define some auxiliary variables
    cdef int i, j, k, idx, it, n
    cdef float rij, rik, loss, deriv
    cdef np.ndarray[np.float32_t, ndim=1] X_i = np.zeros(num_factors, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] Y_j = np.zeros(num_factors, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] Y_k = np.zeros(num_factors, dtype=np.float32)

    #
    # Stochastic Gradient Descent starts here
    #
    for it in range(iters):     # for each iteration
        loss = 0.0
        for n in range(nnz):
            i, j, k = sample[n]
            # get the user and item factors
            X_i = X[i].copy()
            Y_j = Y[j].copy()
            Y_k = Y[k].copy()
            # compute the difference of the predicted scores
            diff_yjk = Y_j - Y_k
            zijk = np.dot(X_i, diff_yjk)
            # compute the sigmoid
            sig = 1. / (1. + exp(-zijk))
            # update the loss
            loss += log(sig)

            # adjust the latent factors
            deriv = 1. - sig
            X[i] += lrate * (deriv * diff_yjk - user_reg * X_i)
            Y[j] += lrate * (deriv * X_i - pos_reg * Y_j)
            Y[k] += lrate * (-deriv * X_i - neg_reg * Y_k)

        loss /= nnz
        if verbose:
            print('Iter {} - loss: {:.4f}'.format(it+1, loss))
        # update the learning rate
        lrate *= lrate_decay
        if use_resampling:
            if sampling_type == 'user_uniform_item_uniform':
                sample = user_uniform_item_uniform_sampling(R, nnz, replace=sample_with_replacement, seed=rnd_seed, verbose=verbose)
            elif sampling_type == 'user_uniform_item_pop':
                sample = user_uniform_item_pop_sampling(R, nnz, alpha=sampling_pop_alpha, seed=rnd_seed, verbose=verbose)

    return X, Y

def user_uniform_item_uniform_sampling(R, size, replace=True, seed=1234, verbose=True):
    # use Cython MemoryViews for fast access to the sparse structure of R
    cdef int [:] col_indices = R.indices, indptr = R.indptr
    cdef int M = R.shape[0], N = R.shape[1]
    cdef int nnz = len(R.data)

    cdef np.ndarray[np.int64_t, ndim=2] sample = np.zeros((size, 3), dtype=np.int64)
    cdef np.ndarray[np.int8_t, ndim=1] is_sampled # boolean arrays are not yet supported by Cython
    if not replace:
        is_sampled = np.zeros(nnz, dtype=np.int8)

    # set the seed of the random number generator
    np.random.seed(seed)

    cdef int i=0, start, end, iid, jid, kid, idx
    cdef np.ndarray[np.int64_t, ndim=1] aux, neg_candidates
    cdef int [:] pos_candidates
    while i < size:
        # 1) sample a user from a uniform distribution
        iid  = np.random.choice(M)

        # 2) sample a positive item uniformly at random
        start = indptr[iid]
        end = indptr[iid+1]
        pos_candidates = col_indices[start:end]
        if start == end:
            # empty candidate set
            continue
        if replace:
            # sample positive items with replacement
            jid = np.random.choice(pos_candidates)
        else:
            # sample positive items without replacement
            # use a index vector between start and end
            aux = np.arange(start, end)
            if np.all(is_sampled[aux]):
                # all positive items have been already sampled
                continue
            idx = np.random.choice(aux)
            while is_sampled[idx]:
                # TODO: remove idx from aux to speed up the sampling
                idx = np.random.choice(aux)
            is_sampled[idx] = 1
            jid = col_indices[idx]

        # 3) sample a negative item uniformly at random
        # build the candidate set of negative items
        # TODO: precompute the negative candidate set for speed-up
        neg_candidates = np.delete(np.arange(N), pos_candidates)
        kid = np.random.choice(neg_candidates)
        sample[i, :] = [iid, jid, kid]
        i += 1
        if verbose and i % 10000 == 0:
            print('Sampling... {:.2f}% complete'.format(i/size*100))
    return sample


def user_uniform_item_pop_sampling(R, size, alpha=1., seed=1234, verbose=True):
    # use Cython MemoryViews for fast access to the sparse structure of R
    cdef int [:] col_indices = R.indices, indptr = R.indptr
    cdef int M = R.shape[0], N = R.shape[1]
    cdef int nnz = len(R.data)

    cdef np.ndarray[np.int64_t, ndim=2] sample = np.zeros((size, 3), dtype=np.int64)

    # compute the item popularity
    cdef np.ndarray[np.float32_t, ndim=1] item_pop = np.asarray(np.sum(R > 0, axis=0)).squeeze().astype(np.float32)
    # smooth popularity with an exponential factor alpha
    item_pop = np.power(item_pop, alpha)

    # set the seed of the random number generator
    np.random.seed(seed)

    cdef int i=0, start, end, iid, jid, kid, idx
    cdef np.ndarray[np.int64_t, ndim=1] aux, neg_candidates
    cdef int [:] pos_candidates
    cdef np.ndarray[np.float32_t, ndim=1] p
    while i < size:
        # 1) sample a user from a uniform distribution
        iid  = np.random.choice(M)

        # 2) sample a positive item proportionally to its popularity
        start = indptr[iid]
        end = indptr[iid+1]
        pos_candidates = col_indices[start:end]
        if start == end:
            # empty candidate set
            continue
        # always sample with replacement
        p = item_pop[pos_candidates]
        p /= np.sum(p)
        jid = np.random.choice(pos_candidates, p=p)

        # 3) sample a negative item uniformly at random
        # build the candidate set of negative items
        # TODO: precompute the negative candidate set for speed-up
        neg_candidates = np.delete(np.arange(N), pos_candidates)
        kid = np.random.choice(neg_candidates)
        sample[i, :] = [iid, jid, kid]
        i += 1
        if verbose and i % 10000 == 0:
            print('Sampling... {:.2f}% complete'.format(i/size*100))
    return sample