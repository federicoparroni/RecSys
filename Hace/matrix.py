import numpy as np
import scipy
from scipy.sparse import csr_matrix

class M:

    def __init__(self):
        #Define the constants
        self.N_PLAYLISTS = 50445+1
        self.N_TRACKS = 20634+1
        self.N_ARTISTS = 6667+1
        self.N_ALBUMS = 12743+1


    def create_urm(self, df):
        urm = np.zeros((self.N_PLAYLISTS, self.N_TRACKS))
        for i in range(df.shape[0]):
            urm[df.iloc[i, 0], df.iloc[i, 1]] = 1
        return urm


    def create_icm(self, df):
        icm = np.zeros((self.N_TRACKS, self.N_ARTISTS + self.N_ALBUMS))
        for i in range(df.shape[0]):
            icm[df.iloc[i, 0], df.iloc[i, 1]] = 1
            icm[df.iloc[i, 0], df.iloc[i, 2]] = 1
        return icm

    ''' create the S_knn matrix from original S

                param_name  | type         | description

        in:     s_matrix    | (csr_matrix) | sparse item similarity matrix in CSR format
        in:     k           | int          | first K nearest neighbour
        -----------------------------------------------------
        out:    sknn_matrix | (csr_matrix) | Sknn matrix

    '''
    def create_Sknn(self, s_matrix, k=-1):
        if k == -1:
            return s_matrix
        else:
            sknn_matrix = scipy.sparse.csr_matrix((s_matrix.shape[0], s_matrix.shape[1]), dtype=np.float64)
            for i in range(0, s_matrix.shape[0]):
                r = s_matrix.getrow(i)
                nonzeros = r.nonzero()
                nonzeros_count = len(nonzeros[1])

                # get the max of each row k times and set that to 0
                for n in range(0, min(nonzeros_count, k)):
                    index = r.argmax()
                    sknn_matrix[i, index] = s_matrix[i, index]
                    r[0, index] = 0

            return sknn_matrix
