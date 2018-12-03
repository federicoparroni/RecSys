import data.data as data
import utils.log as log
import pandas as pd
import math
import numpy as np
from scipy.sparse import csr_matrix

# sequence-aware recommender dataset split
# http://ls13-www.cs.tu-dortmund.de/homepage/publications/jannach/Conference_RecSys_2015_pl.pdf
#

def get_sequences(h):
    """
    Split the dataset of ONLY sequential playlist into train and validation sets

    Parameters
    ----------
    h: (int) history size / length of the sequences

    Returns
    -------
    sequences: (np_array) sequences extracted from the playlists:
        [ [id, track1, track2, ..., trackN], ... ]
    target_indices: (list) indices of the last sequence in the sequences list
    """
    # get sequential playlists dataframe
    seq_playlist_df = pd.DataFrame({'playlist_id': data.get_sequential_target_playlists()})
    seq_train_df = data.get_sequential_train_df().merge(seq_playlist_df)

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

    #return sequences and target_indices
    return np.array(seq, dtype=np.int32), target_indices


""" TEST """
if __name__ == "__main__":
    seq, target_indices = get_sequences(h=3)
