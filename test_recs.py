import data.data as data
import numpy as np
import scipy.sparse as sps
from recommenders.hybrid_base import Hybrid
import utils.log as log
import inout.importexport as inout

base_path = 'raw_data/saved_r_hat_evaluation/'
name = '_slim_rmse_elasticnet_13-05-23.npz'

r_hat_array = []
r_hat_array.append(sps.load_npz(base_path+name))

hybrid_rec = Hybrid(r_hat_array, normalization_mode=Hybrid.MAX_MATRIX, urm_filter_tracks=data.get_urm_train())

rec = hybrid_rec.recommend_batch(weights_array=[1], target_userids=data.get_target_playlists())
rec_seq = hybrid_rec.recommend_batch(weights_array=[1], target_userids=data.get_sequential_target_playlists())
inout.exportcsv(recs=rec_seq)
rec_non_seq = hybrid_rec.recommend_batch(weights_array=[1], target_userids=data.get_target_playlists()[5000:])

log.error('performance on all playlists')
hybrid_rec.evaluate(recommendations=rec, test_urm=data.get_urm_test())

log.error('performance on sequential-playlists')
hybrid_rec.evaluate(recommendations=rec_seq, test_urm=data.get_urm_test())

log.error('performance on non-sequential-playlists')
hybrid_rec.evaluate(recommendations=rec_non_seq, test_urm=data.get_urm_test())

