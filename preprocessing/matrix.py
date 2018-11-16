import numpy as np
import data as d

class M:

    def __init__(self):
        #Define the constants
        self.N_PLAYLISTS = d.N_PLAYLISTS
        self.N_TRACKS = d.N_TRACKS
        self.N_ARTISTS = d.N_ARTISTS
        self.N_ALBUMS = d.N_ALBUMS

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
