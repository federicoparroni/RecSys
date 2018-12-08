"""
Provides a quick access to the dataset and saved matrices
"""

from scipy.sparse import load_npz
import pandas as pd
import scipy.sparse as sps
import numpy as np

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

_URM_TRAIN_PATH = 'raw_data/matrices/urm_train_1.npz'

_URM_TEST_PATH = 'raw_data/matrices/urm_test_1.npz'

_URM_TRAIN_EXPLICIT_PATH = 'raw_data/explicit/urm_train.npz'

_URM_TEST_EXPLICIT_PATH = 'raw_data/explicit/urm_test.npz'

_URM_SEQUENTIAL_MASKED_PATH = 'raw_data/masked_sequential_first_entries/sp_urm_masked_sequential.npz'

_URM_TRAIN_SEQUENTIAL_MASKED_PATH = 'raw_data/masked_sequential_first_entries/sp_urm_masked_sequential_train_MAP.npz'

_URM_TEST_SEQUENTIAL_MASKED_PATH = 'raw_data/masked_sequential_first_entries/sp_urm_masked_sequential_test_MAP.npz'

# sequential dataframes
_SEQUENTIAL_TRAIN_DF_PATH = 'raw_data/matrices/sequential_split/df_train.csv'

_SEQUENTIAL_TEST_DF_PATH = 'raw_data/matrices/sequential_split/df_test.csv'

_ICM_PATH = 'raw_data/matrices/icm.npz'

#sparse_matrices CSR format
_urm = None
_urm_train = None
_urm_test = None
_icm = None
_urm_train_explicit = None
_urm_test_explicit = None

#pandas frame
_tracks_df = None
_playlists_df = None
_seq_train_df = None
_seq_test_df = None

#np.array
_target_playlists = None
_all_playlists = None

#constants
N_PLAYLISTS = 50445 + 1
N_TRACKS = 20634 + 1
N_ARTISTS = 6667 + 1
N_ALBUMS = 12743 + 1

N_SEQUENTIAL = 5000

       
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

def get_urm_train_explicit():
    global _urm_train_explicit
    if _urm_train_explicit is None:
        _urm_train_explicit = load_npz(_URM_TRAIN_EXPLICIT_PATH)
    return _urm_train_explicit

def get_urm_test_explicit():
    global _urm_test_explicit
    if _urm_test_explicit is None:
        _urm_test_explicit = load_npz(_URM_TEST_EXPLICIT_PATH)
    return _urm_test_explicit

def get_urm_sequential_masked():
    global _urm
    if _urm is None:
        _urm = load_npz(_URM_SEQUENTIAL_MASKED_PATH)
    return _urm

def get_urm_train_sequential_masked():
    global _urm_train
    if _urm_train is None:
        _urm_train = load_npz(_URM_TRAIN_SEQUENTIAL_MASKED_PATH)
    return _urm_train

def get_urm_test_sequential_masked():
    global _urm_test
    if _urm_test is None:
        _urm_test = load_npz(_URM_TEST_SEQUENTIAL_MASKED_PATH)
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

def get_sequential_target_playlists():
    return get_target_playlists()[0:N_SEQUENTIAL]

# sequential dataframes
def get_sequential_train_df():
    global _seq_train_df
    if _seq_train_df is None:
        _seq_train_df = pd.read_csv(_SEQUENTIAL_TRAIN_DF_PATH)
    return _seq_train_df

def get_sequential_test_df():
    global _seq_test_df
    if _seq_test_df is None:
        _seq_test_df = pd.read_csv(_SEQUENTIAL_TEST_DF_PATH)
    return _seq_test_df

def get_all_playlists():
    global _all_playlists
    if _all_playlists is None:
        _all_playlists = [p for p in range(N_PLAYLISTS)]
    return _all_playlists

def get_empty_urm():
    empty_urm = sps.csr_matrix((N_PLAYLISTS, N_TRACKS))
    return empty_urm
