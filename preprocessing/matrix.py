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
