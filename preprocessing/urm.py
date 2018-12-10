import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
import os
import time
import pandas as pd
import data.data as d
from random import randint
from preprocessing.process_interactions import ProcessInteractions
from preprocessing.explicate import ExplicateLinearly, ExplicateBase
from preprocessing.split import SplitRandomNonSequentiasLastSequential, SplitRandom

def create_urms(proc_int, explicate, split, save_dataframes=False):
    """
    Creates the full set of files containing sparse matrices needed for the train and test (except for the icm)
    The creation of the urms always starts from the train.csv and is personalized by specifying a ProcessInteractions
    object and a Split object

    @Params
    proc_int        (ProcessInteractions) personalizes the preprocess of the train.csv dataframe
    explicate       (ExplicateBase) specifies the strategy of assigning ratings to tracks
    split           (Split) personalizes the split into train and test of data coming after ProcessInteractions
    save_dataframes (Bool) whether to save the train and test dataframes or not
    """
    path = "raw_data/new" + str(randint(1, 100))
    print('starting dataset creation of urms in ' + path)
    os.mkdir(path)

    # preprocess the interactions, gets the base training dataframe
    start = time.time()
    df = proc_int.process()
    print('interactions preprocessed in: {}'.format(time.time() - start))

    # perform the split, gets the train dataframe
    start = time.time()
    df_train = split.process(df)
    print('split done in: {}'.format(time.time() - start))

    # gets the test dataframe
    df_test = pd.concat([df, df_train]).drop_duplicates(keep=False)

    # assigns ratings to tracks
    start = time.time()
    df_train = explicate.process(df_train)
    b = ExplicateBase()
    df_test = b.process(df_test)
    df = explicate.process(df)
    print('ratings assigned in: {}'.format(time.time() - start))

    if save_dataframes:
        df_train.to_csv('{}/df_train.csv'.format(path), index=False)
        df_test.to_csv('{}/df_test.csv'.format(path), index=False)

    # check if any playlist has at least one element in the test set
    _check_presence_test_samples(df_test, target='target')
    _check_presence_test_samples(df_test, target='all')

    # save matrices
    start = time.time()
    urm = _create_urm(df)
    save_npz(path + '/urm', urm)

    urm_train = _create_urm(df_train)
    save_npz(path + '/urm_train', urm_train)

    urm_test = _create_urm(df_test)
    save_npz(path + '/urm_test', urm_test)
    print('matrices saved in: {}'.format(time.time() - start))



def _check_presence_test_samples(df_test, target='target'):
    """
    Checks that in the test dataframe there is at least one track for each target playlist
    :param df_test: (panda's dataframe)
    :param target_playlists: (list)
    """
    if target == 'all':
        p = d.get_all_playlists()
    elif target == 'target':
        p = d.get_target_playlists()

    if len(df_test[df_test['playlist_id'].isin(p)].groupby('playlist_id')) != len(p):
        if target == 'all':
            print("WARNING: not all the target playlists (JUST THE TARGETS) have a song in the training set")
        elif target == 'target':
            print("WARNING: not all the playlists (ALL OF THEM) have a song in the training set")


def _create_urm(df):
    """
    Utility method
    :param df: (panda's dataframe) represents a set of playlists and tracks (train.csv-like)
    :return:   (csr matrix) the urm built from the df, in format (number of playlists, number of tracks)
    """
    return csr_matrix((df['rating'].values, (df['playlist_id'].values, df['track_id'].values)), 
                      shape=(d.N_PLAYLISTS, d.N_TRACKS))

df = d.get_playlists_df()
pi = ProcessInteractions(df)
s = SplitRandom(0.2)
es = ExplicateBase()
create_urms(pi, es, s, save_dataframes=False)
