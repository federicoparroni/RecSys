import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import load_npz
import implicit
from data import Data
from helpers import model_bridge as bridge
from helpers.export import Export

# load data
d = Data()
targetUsersIds = d.target_playlists_df['playlist_id'].values

# get item_user matrix by transposing the URM matrix
URM = load_npz('../dataset/saved_matrices/sp_urm.npz')
item_user_data = URM.transpose()
print('> data loaded')

# initialize a model (BM25 metric)

model = implicit.nearest_neighbours.BM25Recommender(K=100, K1=0.8, B=0.3)

# train the model on a sparse matrix of item/user/confidence weights
model.fit(item_user_data)

# build recommendations array
recommendations = bridge.array_of_recommendations(model, target_user_ids=targetUsersIds, urm=URM)

# export
Export.export(np.array(recommendations), path='../submissions/', name='collaborative_BM25')
print('file exported')
