import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import load_npz
import implicit
from data import Data
from helpers import model_bridge as bridge
from helpers.manage_dataset.export import Export
import pandas as pd

# load data
targetUsersIds = pd.read_csv('../dataset/target_playlists.csv')['playlist_id'].values

# get item_user matrix by transposing the URM matrix
URM = load_npz('../dataset/saved_matrices/sp_urm_MAP_train.npz')
item_user_data = URM.transpose()
print('> data loaded')

# initialize a model (BM25 metric)

model = implicit.nearest_neighbours.BM25Recommender(K=400, K1=1.5, B=0.3)

# train the model on a sparse matrix of item/user/confidence weights
model.fit(item_user_data)

# build recommendations array
recommendations = bridge.array_of_recommendations(model, target_user_ids=targetUsersIds, urm=URM)

for i in recommendations:
    if len(i) != 11:
        print(a)

# export
Export.export(np.array(recommendations), path='../Hace/submissions/', name='collaborative_BM250063')
print('file exported')
