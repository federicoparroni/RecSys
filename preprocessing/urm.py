import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
import data.data as data as d
from preprocessing.process_interactions import ProcessInteractions
from preprocessing.split import SplitRandomNonSequentiasLastSequential
import os
from random import randint

def create_urms(proc_int, split):
    """
    Creates the full set of files containing sparse matrices needed for the train and test (except for the icm)
    The creation of the urms always starts from the train.csv and is personalized by specifying a ProcessInteractions
    object and a Split object

    @Params
    proc_int        (ProcessInteractions) personalizes the preprocess of the train.csv dataframe
    split           (Split) personalizes the split into train and test of data coming after ProcessInteractions
    """

    path = "raw_data/new" + str(randint(1, 100))
    print('starting dataset creation of urms in ' + path)
    os.mkdir(path)

    # preprocess the interactions
    df = proc_int.process()

    # perform the split
    df_train = split.process(df)

    # create urms
    _save(df, df_train, path)

"""
    sub-methods used by create_urms
"""
def _save(df, df_train, folder_path):
    urm = _create_urm(df)
    sp_urm = csr_matrix(urm)
    save_npz(folder_path + '/urm', sp_urm)

    urm_train = _create_urm(df_train)
    sp_urm_train = csr_matrix(urm_train)
    save_npz(folder_path + '/urm_train', sp_urm)

    sp_urm_MAP = sp_urm - sp_urm_train
    save_npz(folder_path + '/urm_test', sp_urm_MAP)


def _create_urm(df):
    urm = np.zeros((d.N_PLAYLISTS, d.N_TRACKS))
    for i in range(df.shape[0]):
        urm[df.iloc[i, 0], df.iloc[i, 1]] = 1
    return urm


df = d.get_playlists_df()
pi = ProcessInteractions(df)
s = SplitRandomNonSequentiasLastSequential(0.2)
create_urms(pi, s)
