import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import load_npz
import implicit
from data import Data
from helpers import model_bridge as M
from helpers.export import Export

# load data
d = Data()
targetUsersIds = d.target_playlists_df['playlist_id'].values

# get item_user matrix by transposing the URM matrix
URM = load_npz('../dataset/saved_matrices/sp_urm.npz')
item_user_data = URM.transpose()
print('> data loaded')

# initialize a model
model = implicit.als.AlternatingLeastSquares(factors=100, iterations=100)

# train the model on a sparse matrix of item/user/confidence weights
model.fit(item_user_data)

# build recommendations array
recommendations = M.array_of_recommendations(model, target_user_ids=targetUsersIds, urm=URM)

# export
Export.export(np.array(recommendations), path='submissions/')
print("> exported")