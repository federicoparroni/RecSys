import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse import save_npz
import data.data as d
import pandas as pd
import utils.log as log
from preprocessing.process_interactions import ProcessInteractions
from preprocessing.icm import create_icm
from preprocessing.split import SplitRandomNonSequentiasLastSequential
from preprocessing.split import SplitRandom
import os
from random import randint


def create_ucm_from_urm(urm_train):
    """
    Create ucm

    @Params
    proc_int        (ProcessInteractions) personalizes the preprocess of the train.csv dataframe
    split           (Split) personalizes the split into train and test of data coming after ProcessInteractions
    save_dataframes (Bool) whether to save the train and test dataframes or not
    """
    path = "raw_data/ucm" + str(randint(1, 100))
    print('starting dataset creation of UCM in ' + path)

    # maybe can be better a dense array?
    ICM = csr_matrix(create_icm(d.get_tracks_df(), []))
    UCM = lil_matrix((d.N_PLAYLISTS,ICM.shape[1]), dtype=np.int)
    for p in range(d.N_PLAYLISTS):
        track_indices = urm_train[p].nonzero()[1]
        for track_id in track_indices:
            UCM[p] += ICM.getrow(track_id)
        log.progressbar(p, d.N_PLAYLISTS)

    # save matrices
    os.mkdir(path)
    save_npz(path + '/ucm', UCM)

if __name__ == "__main__":    
    urm = d.get_urm_train()
    create_ucm_from_urm(urm_train=urm)
