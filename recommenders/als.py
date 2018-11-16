import numpy as np
from scipy.sparse import load_npz
import implicit
from implicit.evaluation import mean_average_precision_at_k
from data import Data
from recommenders import model_bridge as bridge
from evaluation.map_evaluation import evaluate_map
from helpers.export import Export

# load data
d = Data()
targetUsersIds = d.target_playlists_df['playlist_id'].values

# get item_user matrix by transposing the URM matrix
URM = load_npz('raw_data/matrices/urm.npz')
item_user_data = URM.transpose()
print('> data loaded')

# initialize a model
model = implicit.als.AlternatingLeastSquares(factors=180, iterations=100, regularization=0.1)

# train the model on a sparse matrix of item/user/confidence weights
model.fit(item_user_data)

# build recommendations array
recommendations = bridge.array_of_recommendations(model, target_user_ids=targetUsersIds, urm=URM)

# VALIDATION: load URM test MAP matrix
test_urm = load_npz('raw_data/matrices/sp_urm_test_MAP.npz')
map10 = evaluate_map(recommendations, test_urm)
print('Estimated map --> {}'.format(map10))

map10implicit = mean_average_precision_at_k(model, train_user_items=URM, test_user_items=test_urm)
print('Library evaluation: {}'.format(map10implicit))

# export
Export.export(np.array(recommendations), path='submissions/', name='als')
print("> exported")