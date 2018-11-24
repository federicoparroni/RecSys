from recommenders.hybrid_base import Hybrid
import scipy.sparse as sps
from inout.importexport import exportcsv
from recommenders.hybrid.hybrid_cluster import HybridClusterInteractionsCount
from recommenders.content_based.content_based import ContentBasedRecommender
import data.data as data
import utils.cluster_ensemble as ce
from recommenders.collaborative_filtering.alternating_least_square import AlternatingLeastSquare
from recommenders.collaborative_filtering.pure_SVD import Pure_SVD
from recommenders.collaborative_filtering.userbased import CFUserBased


base_path = 'raw_data/saved_r_hat/'

r_hat_array = []
r_hat_array.append(sps.load_npz(base_path+'slim_bpr_13-05-06.npz'))
r_hat_array.append(sps.load_npz(base_path+'slim_rmse_elasticnet_15-08-30.npz'))
r_hat_array.append(sps.load_npz(base_path+'ALS_18-05-46.npz'))
r_hat_array.append(sps.load_npz(base_path+'CFuser_10-36-03.npz'))
#r_hat_array.append(sps.load_npz(base_path+'pureSVD_19-24-20.npz'))

print('matrices loaded')

weights = [74, 77, 77, 74]

hybrid_rec = Hybrid(r_hat_array)


recommendations = hybrid_rec.recommend_batch(weights_array=weights)

exportcsv(recommendations, path='submissions', name='hybrid')


