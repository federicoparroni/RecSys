import data.data as data
import utils.log as log
import pandas as pd
import math
import numpy as np
from scipy.sparse import csr_matrix

# sequence-aware recommender dataset split
# http://ls13-www.cs.tu-dortmund.de/homepage/publications/jannach/Conference_RecSys_2015_pl.pdf
#

def preprocess(h, split_perc=0.2):
    """
    Split the dataset of ONLY sequential playlist into train and validation sets

    Parameters
    ----------
    h: (int) history size / length of the sequences
    split_perc: (float) validation split percentage, 0 to skip the creation of the validation set

    Returns
    -------
    urm_train: (sparse) csr matrix of shape (n_playlist,n_tracks)
    urm_test: (sparse) csr matrix of shape (n_playlist,n_tracks)
    sequences: (np_array) sequences extracted from the playlists:
        [ [id, track1, track2, ..., trackN], ... ]
    target_indices: (list) indices of the last sequence in the sequences list
    """
    if split_perc > 1 or split_perc < 0:
        log.error('Invalid split percentage: {}'.format(split_perc))

    train_df = data.get_playlists_df()
    seq_df = data.get_sequential_target_playlists_df()
    seq_df = train_df.merge(seq_df)

    # split train and validation
    seq_train_df = seq_df.groupby('playlist_id').apply(
            lambda x: x.iloc[:-math.floor(len(x) * split_perc)]).reset_index(drop=True)
    seq_validation_df = pd.concat([seq_df, seq_train_df]).drop_duplicates(keep=False)

    # build sequences of h tracks: seq
    grouped = seq_train_df.groupby('playlist_id')
    seq = []
    target_indices = []
    for id, group_df in grouped:
        for i in range(group_df.shape[0]-h):
            s = group_df['track_id'].values[i:i+h]
            row = np.concatenate(([id],s))
            seq.append(row)
        target_indices.append(len(seq)-1)

    # build URM_train
    urm_train = csr_matrix((data.N_PLAYLISTS,data.N_TRACKS), dtype=np.int32)
    urm_train[(seq_train_df['playlist_id'].values, seq_train_df['track_id'].values)] = 1

    # build URM_test
    urm_test = csr_matrix((data.N_PLAYLISTS,data.N_TRACKS), dtype=np.int32)
    urm_test[(seq_validation_df['playlist_id'].values, seq_validation_df['track_id'].values)] = 1

    #return seq, seq_validation_df, target_indices #, urm_train, urm_test
    return urm_train, urm_test, np.array(seq, dtype=np.int32), target_indices


""" TEST """
if __name__ == "__main__":
    urm_train, urm_test, seq, target_indices = preprocess(h=3)
