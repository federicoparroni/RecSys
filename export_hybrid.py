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

base_path = 'raw_data/saved_r_hat/'

start = time.time()

r_hat_array = []

r_hat_array.append(sps.load_npz(base_path+'ALS_20-26-44.npz'))
r_hat_array.append(sps.load_npz(base_path+'userKNN.npz'))
r_hat_array.append(sps.load_npz(base_path+'4collaborative_2.npz'))

print('matrices loaded in {:.2f} s'.format(time.time() - start))

weights = [1, 0, 1]

hybrid_rec = HybridRHat(r_hat_array, normalization_mode=Hybrid.L2, urm_filter_tracks=data.get_urm())

#sps.save_npz(base_path+'', hybrid_rec.get_r_hat(weights_array=[ 0.6573, 0.1448]))

recommendations = hybrid_rec.recommend_batch(weights_array=weights, target_userids=data.get_target_playlists())

exportcsv(recommendations, path='submissions', name='hybrid')