from recommenders.hybrid_base import Hybrid
from recommenders.hybrid_r_hat import HybridRHat
from recommenders.hybrid_similarity import HybridSimilarity
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

r_hat_array.append(sps.load_npz(base_path+'slim_bpr_13-05-06.npz'))
r_hat_array.append(sps.load_npz(base_path+'slim_rmse_elasticnet_20-08-49.npz'))
r_hat_array.append(sps.load_npz(base_path+'ALS_20-26-44.npz'))
r_hat_array.append(sps.load_npz(base_path+'_content_based_14-15-56.npz'))
#r_hat_array.append(sps.load_npz(base_path+'pureSVD_19-24-20.npz'))
#r_hat_array.append(sps.load_npz(base_path+'CFuser_10-36-03.npz'))
r_hat_array.append(sps.load_npz(base_path+'userKNN.npz'))
r_hat_array.append(sps.load_npz(base_path+'cos+sim_item.npz'))
#r_hat_array.append(sps.load_npz(base_path+'CFitem_13-32-53.npz'))
#r_hat_array.append(sps.load_npz(base_path+'CFitem_cosine_15-04-40.npz'))

#low, high = cluster.cluster_users_by_interactions_count(10)

print('matrices loaded')



weights = [1, 1, 1, 1, 1, 1]

hybrid_rec = Hybrid(r_hat_array, normalization_mode=Hybrid.MAX_MATRIX, urm_filter_tracks=data.get_urm())

#sps.save_npz(base_path+'cos+sim_item', hybrid_rec.get_r_hat(weights_array=[ 0.6573, 0.1448]))
recommendations = hybrid_rec.recommend_batch(weights_array=weights, target_userids=data.get_target_playlists())

exportcsv(recommendations, path='submissions', name='hybrid')
"""

"""
#=========== USE THIS PART OF CODE TO EVALUATE HYBRID ===============
start = time.time()

base_path = 'raw_data/saved_r_hat_evaluation/'

r_hat_array = []

r_hat_array.append(sps.load_npz(base_path+'_slim_bpr_18-42-00.npz'))
r_hat_array.append(sps.load_npz(base_path+'_ALS_13-29-41.npz'))
r_hat_array.append(sps.load_npz(base_path+'_slim_rmse_elasticnet_13-05-23.npz'))
r_hat_array.append(sps.load_npz(base_path+'_content_based_17-58-51.npz'))
r_hat_array.append(sps.load_npz(base_path+'userKNN.npz'))
r_hat_array.append(sps.load_npz(base_path+'cos+sim_item.npz'))


#r_hat_array.append(sps.load_npz(base_path+'COSINE_CFitem_14-25-51.npz'))
#r_hat_array.append(sps.load_npz(base_path+'_CFitem_13-31-58.npz'))


print('MATRICES LOADED')
print('{:.2f}'.format(time.time() - start))

hybrid_rec = HybridRHat(r_hat_array, normalization_mode=Hybrid.MAX_MATRIX, urm_filter_tracks=data.get_urm_train())

recs = hybrid_rec.recommend_batch(weights_array=[1,1,1,1,1,1], target_userids=data.get_target_playlists())

hybrid_rec.evaluate(recs, test_urm=data.get_urm_test())
#sps.save_npz(base_path+'cos+sim_item',hybrid_rec.get_r_hat(weights_array=[ 0.6573, 0.1448]))

#print(hybrid_rec.validate(iterations=500, urm_test=data.get_urm_test(),userids=data.get_target_playlists()))
"""


base_path = 'raw_data/saved_sim_matrix_evaluation/'

sim_matrices_array = []

sim_matrices_array.append(sps.load_npz(base_path+'slim_rmse.npz'))
sim_matrices_array.append(sps.load_npz(base_path+'content_based.npz'))
sim_matrices_array.append(sps.load_npz(base_path+'CF_ITEM_SIM_SPLUS.npz'))

hybrid_rec = HybridSimilarity(sim_matrices_array, normalization_mode=HybridSimilarity.MAX_ROW, urm_filter_tracks=data.get_urm_train())

#recs = hybrid_rec.recommend_batch(weights_array=[1,1,1], target_userids=data.get_target_playlists())
#hybrid_rec.evaluate(recommendations=recs, test_urm=data.get_urm_test())
hybrid_rec.validate(iterations=100, urm_test=data.get_urm_test(), userids=data.get_target_playlists())


