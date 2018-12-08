from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt
import data.data as d
from random import randint
import os
from scipy.sparse import save_npz

def create_icm(df, thr):
    """
    code needed to create the icm: matrix #tracks x #artists + #albums + length(thr)+1
    for any track, we will have set to 1 the columns associated with the artists and the track
    and to the right threshold value of duration
    :param df: (panda's dataframe)
    :param thr: (list) threshold for durations. eg: - [[0 1.5], [2.5 4.5]] will cluster the matrix tracks in the clusters:
                                                    [0, 1.5], [2.5, 4.5]
                                                    - [] will not cluster at all
    :return: icm
    """
    icm = np.zeros((d.N_TRACKS, d.N_ALBUMS + d.N_ARTISTS + len(thr)))
    for i in range(df.shape[0]):
        icm[df.iloc[i, 0], df.iloc[i, 1]] = 1               # weight for album
        icm[df.iloc[i, 0], d.N_ALBUMS + df.iloc[i, 2]] = 1  # weight for artist
        for j in range(len(thr)):
            duration = df.iloc[i, 3]/60
            if duration >= thr[j][0] and duration < thr[j][1]:
                icm[df.iloc[i, 0], d.N_ARTISTS + d.N_ALBUMS + j] = 1
                break
    return icm

if __name__ == "__main__":
    icm = create_icm(d.get_tracks_df(), [[0, 0.75], [0.75, 1.5], [8, 12], [12, np.inf]])
    icm = csr_matrix(icm)
    path = "raw_data/new" + str(randint(1, 100))
    print('starting dataset creation of icm in ' + path)
    os.mkdir(path)
    save_npz(path + '/icm', icm)

# df = d.get_tracks_df()
# df['duration_sec'] = df['duration_sec']/60
# df.hist(column='duration_sec', bins=150)
# plt.xticks(np.arange(0, 15, 1))
# plt.yticks(np.arange(0, 3000, 200))
# plt.show()