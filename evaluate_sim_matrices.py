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

base_path = 'raw_data/saved_sim_matrix/'

start = time.time()

sim_matrices_array = []

sim_matrices_array.append(sps.load_npz(base_path+'slim_rmse.npz'))
sim_matrices_array.append(sps.load_npz(base_path+'content_based.npz'))
sim_matrices_array.append(sps.load_npz(base_path+'CF_ITEM_SIM_SPLUS.npz'))
sim_matrices_array.append(sps.load_npz(base_path+'slim_bpr.npz'))

print('matrices loaded in {:.2f} s'.format(time.time() - start))

weights = [0.9986, 0.6153, 0.8733, 0.9836]
normalization_mode = HybridSimilarity.L2

hybrid_rec = HybridSimilarity(sim_matrices_array, normalization_mode=normalization_mode, urm_filter_tracks=data.get_urm_train_1())
#sps.save_npz('raw_data/saved_r_hat/4collaborative_2',hybrid_rec.get_r_hat(weights_array=weights))

recs = hybrid_rec.recommend_batch(weights_array=weights, target_userids=data.get_target_playlists())
hybrid_rec.evaluate(recommendations=recs, test_urm=data.get_urm_test_1())

#hybrid_rec.validate(iterations=100, urm_test=data.get_urm_test(), userids=data.get_target_playlists())