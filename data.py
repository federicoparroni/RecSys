"""
Provides a quick access to the dataset and saved matrices
"""

from scipy.sparse import load_npz
import pandas as pd

""" 
PATHS 
"""
# || track_id || album_id || artist_id || duration_sec
_TRACKS_PATH = 'raw_data/original_csv/tracks.csv'

# || playlist_id || track_id ||
_PLAYLISTS_PATH = 'raw_data/original_csv/train.csv'

# || playlist_id ||
_TARGET_PLAYLISTS_PATH = 'raw_data/original_csv/target_playlists.csv'

_URM_PATH = 'raw_data/matrices/urm.npz'

_URM_TRAIN_PATH = 'raw_data/matrices/urm_train.npz'

_URM_TEST_PATH = 'raw_data/matrices/urm_test.npz'

_ICM_PATH = 'raw_data/matrices/icm.npz'


#sparse_matrices CSR format
_urm = None
_urm_train = None
_urm_test = None
_icm = None

#pandas frame
_tracks_df = None
_playlists_df = None

#np.array
_target_playlists = None
_all_playlists = None

#constants
N_PLAYLISTS = 50445 + 1
N_TRACKS = 20634 + 1
N_ARTISTS = 6667 + 1
N_ALBUMS = 12743 + 1
       
""" 
GET METHODS 
"""
    
def get_urm():
    global _urm
    if _urm is None:
        _urm = load_npz(_URM_PATH)
    return _urm

def get_urm_train():
    global _urm_train
    if _urm_train is None:
        _urm_train = load_npz(_URM_TRAIN_PATH)
    return _urm_train

def get_urm_test():
    global _urm_test
    if _urm_test is None:
        _urm_test = load_npz(_URM_TEST_PATH)
    return _urm_test

def get_icm():
    global _icm
    if _icm is None:
        _icm = load_npz(_ICM_PATH)
    return _icm

def get_tracks_df():
    global _tracks_df
    if _tracks_df is None:
        _tracks_df = pd.read_csv(_TRACKS_PATH)
    return _tracks_df

def get_playlists_df():
    global _playlists_df
    if _playlists_df is None:
        _playlists_df = pd.read_csv(_PLAYLISTS_PATH)
    return _playlists_df

def get_target_playlists():
    global _target_playlists
    if _target_playlists is None:
        _target_playlists = [p[0] for p in pd.read_csv(_TARGET_PLAYLISTS_PATH).values]
    return _target_playlists

def get_all_playlists():
    global _all_playlists
    if _all_playlists is None:
        _all_playlists = [p for p in range(N_PLAYLISTS)]
    return _all_playlists
