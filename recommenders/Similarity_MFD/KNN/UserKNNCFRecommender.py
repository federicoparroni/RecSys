
import utils.check_matrix_format as cm
from recommenders.Similarity_MFD.SimilarityMatrixRecommender import SimilarityMatrixRecommender
from recommenders.Similarity_MFD.Compute_Similarity import Compute_Similarity
import data.data as data
from recommenders.Similarity_MFD.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
import numpy as np
import utils.log as log
import scipy.sparse as sps

class UserKNNCFRecommender(SimilarityMatrixRecommender):
    """ UserKNN recommender"""

    RECOMMENDER_NAME = "UserKNNCFRecommender"

    def __init__(self, URM_train, sparse_weights=True):
        super(UserKNNCFRecommender, self).__init__()

        # Not sure if CSR here is faster
        self.URM_train = cm.check_matrix(URM_train, 'csr')

        self.dataset = None

        self.sparse_weights = sparse_weights

        self.compute_item_score = self.compute_score_user_based



    def fit(self, topK=150, shrink=10, similarity='tversky', normalize=True, **similarity_args):

        self.topK = topK
        self.shrink = shrink

        similarity = Compute_Similarity(self.URM_train.T, shrink=shrink, topK=topK, normalize=normalize,
                                        similarity=similarity, **similarity_args)

        if self.sparse_weights:
            self.W_sparse = similarity.compute_similarity()
        else:
            self.W = similarity.compute_similarity()
            self.W = self.W.toarray()



rec = UserKNNCFRecommender(URM_train=data.get_urm_train())
rec.fit()

r_hat = rec.W_sparse.dot(data.get_urm_train())
rec2 = ItemKNNCFRecommender(r_hat)
rec2.fit()
recs = rec2.recommend_batch(userids=data.get_target_playlists(), type='ITEM', filter_seen_matrix=data.get_urm_train())
rec2.evaluate(recs, data.get_urm_test())

#recs = rec.recommend_batch(userids=data.get_target_playlists(), type='USER')
#rec.evaluate(recs, test_urm=data.get_urm_test())
