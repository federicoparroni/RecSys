import numpy as np
import implicit
from scipy.sparse import load_npz
from data import Data
from helpers import model_bridge as bridge
from helpers.manage_dataset.export import Export
from election_methods import ElectionMethods
from evaluation import map_evaluation

# load data
d = Data()
targetUsersIds = d.target_playlists_df['playlist_id'].values

# get item_user matrix by transposing the URM matrix and convert it to COO
URM = load_npz('../dataset/saved_matrices/sp_urm_train_MAP.npz')
test_urm = load_npz('../dataset/saved_matrices/sp_urm_test_MAP.npz')


item_user_data = URM.transpose().tocoo()
print('> data loaded')

# initialize a model
model = implicit.bpr.BayesianPersonalizedRanking(factors=100, iterations=1000, learning_rate=0.1)

# train the model on a sparse matrix of item/user/confidence weights
model.fit(item_users=item_user_data)

# build recommendations array
recommendations = bridge.array_of_recommendations(model, target_user_ids=targetUsersIds, urm=URM)


map = map_evaluation.evaluate_map(recommendations, test_urm)
print('estimated map:',  map)


# export
Export.export(np.array(recommendations), path='../submissions/', name='bayesian_ranking')
print("> exported")
