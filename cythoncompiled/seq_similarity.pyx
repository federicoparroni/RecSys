from __future__ import print_function
import data.data as data
from scipy.sparse import csr_matrix
import numpy as np
from cython.view cimport array as cvarray

# return 1 if sequence hprime contains t, else 0
cpdef int seq_contains(int[:] hprime, int t) nogil:
    cdef int i = 0
    for i in range(hprime.shape[0]):
        if hprime[i] == t:
            return 1
    return 0

"""
# compute binary cosine similarity measure: number common elements / tot elements
cpdef float sim_cosine(int[:] h, int[:] hprime):
    cdef int common_elements = 0
    cdef int slen = len(h)
    for i in range(slen):
        for j in range(slen):
            if h[i] == hprime[j]:
                common_elements += 1
    return common_elements / slen

cpdef float[:,:] compute_scores(int[:,:] sequences, int[:] targetrows, int knn=10):
    cdef int NS = sequences.shape[0]
    cdef int NT = data.N_TRACKS
    cdef float[:,:] scores = np.zeros((NS,NT), dtype=np.float32)
    #scores = csr_matrix((NS,NT), dtype=np.float)
    # compute scores for all sequence i and next track t
    cdef int h, t, hprime
    cdef int[:] h_, hprime_
    for h in range(NS):
        for t in range(NT):
            scores[h,t] = 0
            h_ = sequences[h,1:]    # skip first element (playlist id)
            for hprime in range(NS):
                hprime_ = sequences[hprime,1:]
                if h != hprime:   # skip similarity with the self same sequence
                    scores[h,t] += sim_cosine(h_,hprime_) if seq_contains(hprime_,t) else 0
            if scores[h,t] > 0:
                print("SI")
        print('{}/{}'.format(h,NS))
    return scores
"""


cpdef getH(int[:,:] sequences):
    cdef int NS = sequences.shape[0]
    cdef int printNS = NS-1
    cdef int NT = data.N_TRACKS
    cdef float[:,:] H = np.zeros((NS,NT), dtype=np.float32)
    #H = csr_matrix((NS,NT), dtype=np.float32)
    
    cdef int h, t, track_id
    cdef int[:] seq_h
    for h in range(NS):
        seq_h = sequences[h,1:]    # skip first element (playlist id)
        for t in range(len(seq_h)):
            track_id = seq_h[t]
            H[h,track_id] = 1
        
        print('{}/{}'.format(h,printNS), end='\r')
    print()
    print('Converting H to sparse...')
    return csr_matrix(H)
    #return np.array(H)
