import numpy as np
import implicit
from scipy.sparse import load_npz
from data import Data
from helpers import model_bridge as M
from helpers.export import Export

# load data
d = Data()
targetUsersIds = d.target_playlists_df['playlist_id'].values
# targetUsersIds = d.all_playlists['playlist_id'].values

URM = load_npz('../dataset/saved_matrices/sp_urm.npz')
item_user_data = URM.transpose().tocoo()
print('> data loaded')

# initialize a model
model = implicit.bpr.BayesianPersonalizedRanking(factors=300, iterations=100, learning_rate=0.01)

# train the model on a sparse matrix of item/user/confidence weights
model.fit(item_users=item_user_data)

# build recommendations array
recommendations = M.array_of_recommendations(model, target_user_ids=targetUsersIds, urm=URM)

# export
Export.export(np.array(recommendations), path='../Hace/submissions/')
print("> exported")
