import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz

class M:

    # TODO: drop class and make it as a file

    def __init__(self):
        # define the constants
        self.N_PLAYLISTS = 50445+1
        self.N_TRACKS = 20634+1
        self.N_ARTISTS = 6667+1
        self.N_ALBUMS = 12743+1

    def _create_urm(self, df):
        urm = np.zeros((self.N_PLAYLISTS, self.N_TRACKS))
        for i in range(df.shape[0]):
            urm[df.iloc[i, 0], df.iloc[i, 1]] = 1
        return urm

    def _create_icm(self, df):
        icm = np.zeros((self.N_TRACKS, self.N_ARTISTS + self.N_ALBUMS))
        for i in range(df.shape[0]):
            icm[df.iloc[i, 0], df.iloc[i, 1]] = 1
            icm[df.iloc[i, 0], df.iloc[i, 2]] = 1
        return icm

    def save(self, df, df_train, folder_path):
        urm = self._create_urm(df)
        sp_urm = csr_matrix(urm)
        save_npz(folder_path + 'urm', sp_urm)

        urm_train = self._create_urm(df_train)
        sp_urm_train = csr_matrix(urm_train)
        save_npz(folder_path + 'urm_train', sp_urm)

        sp_urm_MAP = sp_urm - sp_urm_train
        save_npz(folder_path + 'urm_test', sp_urm_MAP)



    # creating icm removing the duration feature and save to a file

    # icm = m.create_icm(d.tracks_df.filter(items=['track_id', 'album_id', 'artist_id']))
    # sp_icm = csr_matrix(icm)
    # save_npz('../raw_data/matrices/sp_icm', sp_icm)