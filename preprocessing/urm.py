import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
import data as d
from preprocessing.urm import save
from preprocessing.process_interactions import ProcessInteractions
from preprocessing.split import SplitRandom
import pandas as pd
import os


def create_urm(df):
    urm = np.zeros((d.N_PLAYLISTS, d.N_TRACKS))
    for i in range(df.shape[0]):
        urm[df.iloc[i, 0], df.iloc[i, 1]] = 1
    return urm


def save(df, df_train, folder_path):
    urm = create_urm(df)
    sp_urm = csr_matrix(urm)
    save_npz(folder_path + 'urm', sp_urm)

    urm_train = create_urm(df_train)
    sp_urm_train = csr_matrix(urm_train)
    save_npz(folder_path + 'urm_train', sp_urm)

    sp_urm_MAP = sp_urm - sp_urm_train
    save_npz(folder_path + 'urm_test', sp_urm_MAP)


def create_urms(proc_int, split):
    # preprocess the interactions
    df = proc_int.process()

    # perform the split
    df_train = split.process(df)

    path = "../raw_data/new"
    os.mkdir(path)

    # create urms
    save(df, df_train, path)


df = pd.read_csv('../raw_data/orginal_csv/train.csv')
pi = ProcessInteractions(df)
s = SplitRandom(0.2)
create_urms(pi, s)
