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

start = time.time()

base_path = 'raw_data/saved_r_hat_evaluation/'

r_hat_array = []

#SELECT MODEL TO HYBRID
#r_hat_array.append(sps.load_npz(base_path+'_slim_bpr_18-42-00.npz'))
#r_hat_array.append(sps.load_npz(base_path+'_ALS_13-29-41.npz'))
#r_hat_array.append(sps.load_npz(base_path+'_slim_rmse_elasticnet_13-05-23.npz'))
#r_hat_array.append(sps.load_npz(base_path+'_content_based_17-58-51.npz'))
#r_hat_array.append(sps.load_npz(base_path+'userKNN.npz'))
#r_hat_array.append(sps.load_npz(base_path+'cos+sim_item.npz'))
#r_hat_array.append(sps.load_npz(base_path+'4collaborative_2.npz'))
#r_hat_array.append(sps.load_npz(base_path+'COSINE_CFitem_14-25-51.npz'))
#r_hat_array.append(sps.load_npz(base_path+'_CFitem_13-31-58.npz'))
#r_hat_array.append(sps.load_npz(base_path+'_pureSVD_15-00-26.npz'))

print('matrices loaded in {:.2f} s'.format(time.time() - start))

weights = []

hybrid_rec = HybridRHat(r_hat_array, normalization_mode=Hybrid.L2, urm_filter_tracks=data.get_urm_train_1())

#sps.save_npz(base_path+'cos+sim_item',hybrid_rec.get_r_hat(weights_array=[ 0.6573, 0.1448]))

recs = hybrid_rec.recommend_batch(weights_array=[1,0.03,1], target_userids=data.get_target_playlists())
hybrid_rec.evaluate(recs, test_urm=data.get_urm_test())


#print(hybrid_rec.validate(iterations=500, urm_test=data.get_urm_test_1(), userids=data.get_target_playlists()))


