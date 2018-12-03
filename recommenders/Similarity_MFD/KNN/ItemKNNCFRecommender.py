
import utils.check_matrix_format as cm
from recommenders.Similarity_MFD.SimilarityMatrixRecommender import SimilarityMatrixRecommender
import data.data as data

from recommenders.Similarity_MFD.Compute_Similarity import Compute_Similarity
from recommenders.Similarity_MFD.Compute_Similarity import SimilarityFunction


class ItemKNNCFRecommender(SimilarityMatrixRecommender):
    """ ItemKNN recommender"""

    RECOMMENDER_NAME = "ItemKNNCFRecommender"

    def __init__(self, URM_train, sparse_weights=True):
        super(ItemKNNCFRecommender, self).__init__()

        # CSR is faster during evaluation
        self.URM_train = cm.check_matrix(URM_train, 'csr')

        self.dataset = None

        self.sparse_weights = sparse_weights

    def fit(self, topK=400, shrink=100, similarity=SimilarityFunction.TANIMOTO.value, normalize=True, **similarity_args):

        self.topK = topK
        self.shrink = shrink

        similarity = Compute_Similarity(self.URM_train, shrink=shrink, topK=topK, normalize=normalize, similarity = similarity, **similarity_args)


        if self.sparse_weights:
            self.W_sparse = similarity.compute_similarity()
        else:
            self.W = similarity.compute_similarity()
            self.W = self.W.toarray()

rec = ItemKNNCFRecommender(URM_train=data.get_urm_train())
rec.fit()
recs = rec.recommend_batch(userids=data.get_target_playlists(), type='ITEM')
rec.evaluate(recs, test_urm=data.get_urm_test())

