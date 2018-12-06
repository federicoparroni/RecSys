from recommenders.hybrid_base import Hybrid
import scipy.sparse as sps
import numpy as np
from inout.importexport import exportcsv
from recommenders.hybrid.hybrid_cluster import HybridClusterInteractionsCount
from recommenders.content_based.content_based import ContentBasedRecommender
import data.data as data
import utils.cluster_ensemble as ce
from recommenders.collaborative_filtering.alternating_least_square import AlternatingLeastSquare
from recommenders.collaborative_filtering.pure_SVD import Pure_SVD
from recommenders.collaborative_filtering.userbased import CFUserBased
from recommenders.collaborative_filtering.SLIM_RMSE import SLIMElasticNetRecommender
from recommenders.collaborative_filtering.itembased import CFItemBased
from recommenders.content_based.content_based import ContentBasedRecommender
import os
import time
import clusterize.cluster as cluster


"""
base_path = 'raw_data/saved_r_hat/'

r_hat_array = []
#r_hat_array.append(sps.load_npz(base_path+'sequential_masked_slim_bpr_22-42-07.npz'))
#r_hat_array.append(sps.load_npz(base_path+'sequential_masked_slim_rmse_elasticnet_22-31-54.npz'))
#r_hat_array.append(sps.load_npz(base_path+'sequential_masked_ALS_22-06-57.npz'))
#r_hat_array.append(sps.load_npz(base_path+'sequential_masked_CFitem_22-59-50.npz'))

r_hat_array.append(sps.load_npz(base_path+'slim_bpr_13-05-06.npz'))
r_hat_array.append(sps.load_npz(base_path+'slim_rmse_elasticnet_20-08-49.npz'))
r_hat_array.append(sps.load_npz(base_path+'ALS_20-26-44.npz'))
#r_hat_array.append(sps.load_npz(base_path+'_CFitem_22-51-56.npz'))
r_hat_array.append(sps.load_npz(base_path+'_content_based_14-15-56.npz'))
#r_hat_array.append(sps.load_npz(base_path+'pureSVD_19-24-20.npz'))
#r_hat_array.append(sps.load_npz(base_path+'CFuser_10-36-03.npz'))
r_hat_array.append(sps.load_npz(base_path+'userKNN.npz'))

#low, high = cluster.cluster_users_by_interactions_count(10)

print('matrices loaded')

weights = [72, 81, 76, 32, 71]

hybrid_rec = Hybrid(r_hat_array, normalization_mode=Hybrid.MAX_MATRIX)


recommendations = hybrid_rec.recommend_batch(weights_array=weights)

exportcsv(recommendations, path='submissions', name='hybrid')
"""
#=========== USE THIS PART OF CODE TO EVALUATE HYBRID ===============
start = time.time()

base_path = 'raw_data/saved_r_hat_evaluation/'

r_hat_array = []

r_hat_array.append(sps.load_npz(base_path+'_slim_bpr_18-42-00.npz'))
#r_hat_array.append(sps.load_npz(base_path+'CFuser_22-32-05.npz'))
#r_hat_array.append(sps.load_npz(base_path+'slim_rmse_elasticnet_01-57-50.npz'))

print('MATRICES LOADED')
print('{:.2f}'.format(time.time() - start))

hybrid_rec = Hybrid(r_hat_array, normalization_mode=Hybrid.MAX_MATRIX, urm_filter_tracks=data.get_urm_train())

recs = hybrid_rec.recommend_batch(weights_array=[1], target_userids=data.get_target_playlists())
hybrid_rec.evaluate(recs, test_urm=data.get_urm_test())

#print(hybrid_rec.validate(iterations=100, urm_test=data.get_urm_test()))


