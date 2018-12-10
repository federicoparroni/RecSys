import numpy as np
from scipy.sparse import load_npz
import implicit
import pandas as pd
import data.data as data
import scipy.sparse as sps

# load data
targetUsersIds = data.get_target_playlists()

# get item_user matrix by transposing the URM matrix
URM = data.get_urm_train_1()
item_user_data = URM.transpose()
print('> data loaded')

# initialize a model (BM25 metric)
model = implicit.nearest_neighbours.BM25Recommender(K=400, K1=1.5, B=0.3)

# train the model on a sparse matrix of item/user/confidence weights
model.fit(item_user_data)

r_hat = np.dot(URM[targetUsersIds], model.similarity)
sps.save_npz('raw_data/saved_r_hat_evaluation/BM25', r_hat)

"""
# build recommendations array
recommendations = bridge.array_of_recommendations(model, target_user_ids=targetUsersIds, urm=URM)

Export.export(np.array(recommendations), path='../submissions/', name='BM25 K {} K1 {} B{}'.format(K, K1, B))

print('file exported')


# export with scores
#Export.export_with_scores(recommendations, path='../submissions/', name='BM25 K {} K1 {} B {}'.format(K, K1, B))
#print('file exported')
"""