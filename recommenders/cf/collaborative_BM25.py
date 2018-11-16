import numpy as np
from scipy.sparse import load_npz
import implicit
from implicit.evaluation import mean_average_precision_at_k
from evaluation.map_evaluation import evaluate_map
from recommenders import model_bridge as bridge
from io.export_rec import Export
import pandas as pd


# load data
targetUsersIds = pd.read_csv('../raw_data/target_playlists.csv')['playlist_id'].values

# get item_user matrix by transposing the URM matrix
URM = load_npz('../raw_data/matrices/urm_train.npz')
item_user_data = URM.transpose()
print('> data loaded')

# initialize a model (BM25 metric)
model = implicit.nearest_neighbours.BM25Recommender(K=400, K1=1.5, B=0.3)

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

Export.export(np.array(recommendations), path='../submissions/', name='BM25 K {} K1 {} B{}'.format(K, K1, B))

print('file exported')


# export with scores
#Export.export_with_scores(recommendations, path='../submissions/', name='BM25 K {} K1 {} B {}'.format(K, K1, B))
#print('file exported')
