import pandas as pd
import numpy as np
import scipy
from scipy.sparse import csr_matrix

# test
# from scipy.sparse import load_npz

#===============================================


# || track_id || album_id || artist_id || duration_sec
tracks_path = '../dataset/tracks.csv'

# || playlist_id || track_id ||
playlists_path = '../dataset/train.csv'

# || playlist_id ||
target_playlists_path = '../dataset/target_playlists.csv'


class Data:

    def __init__(self):

        #creating dataframe with pandas
        self.tracks_df = pd.read_csv(tracks_path)
        self.tracks_df.columns = ['track_id', 'album_id', 'artist_id', 'duration_sec']

        self.playlists_df = pd.read_csv(playlists_path)
        self.playlists_df.columns = ['playlist_id', 'track_id']

        self.target_playlists_df = pd.read_csv(target_playlists_path)

        self.unified_df = self.playlists_df.merge(self.tracks_df, on='track_id')

        self.tg_pl_df_m = pd.merge(self.target_playlists_df, self.unified_df)

    def URM(self):
        return self.playlists_df.groupby('playlist_id').agg(''.join)


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
    def create_Sknn(self, s_matrix, k=50):
        sknn_matrix = scipy.sparse.csr_matrix((self.N_TRACKS, self.N_TRACKS), dtype=np.float64)
        for i in range(0, self.N_TRACKS):
            r = s_matrix.getrow(i)
            nonzeros = r.nonzero()
            nonzeros_count = len(nonzeros[1])

            # get the max of each row k times and set that to 0
            for n in range(0, min(nonzeros_count, k)):
                index = r.argmax()
                sknn_matrix[i, index] = s_matrix[i, index]
                r[0, index] = 0

        return sknn_matrix

# test
'''
sp_icm = load_npz('../dataset/saved_matrices/sp_icm.npz')
sp_sim_matrix = sp_icm * sp_icm.transpose()

m = M()
sknn = m.create_Sknn(sp_icm, k=5)

for i in range(0, 20634+1):
    r = sknn.getrow(i)
    nonzeros = r.nonzero()
    nonzeros_count = len(nonzeros[1])
    if nonzeros_count > 1:
        print('error! number of nonzeros: {}'.format(nonzeros_count))

print("success")
'''
