
import utils.check_matrix_format as cm
from recommenders.Similarity_MFD.SimilarityMatrixRecommender import SimilarityMatrixRecommender
import data.data as data
import sys
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

# from scipy.sparse import load_npz
# train = load_npz('raw_data/explicit/urm_train.npz')
# test = load_npz('raw_data/explicit/urm_test.npz')
# rec = ItemKNNCFRecommender(URM_train=train)

# for k in [2200, 2500, 2800, 3000]:
#     for s in [10, 25, 50, 75, 100]:
#         for sim in ["cosine", "adjusted", "pearson", "jaccard", "tanimoto"]:
#             print('knn: {} shrink: {} similarity: {}'.format(k, s, sim))
#             rec.fit(shrink=s, topK=k, similarity=sim)
#             recs = rec.recommend_batch(userids=data.get_sequential_target_playlists(), type='ITEM')
#             rec.evaluate(recs, test_urm=test)

# rec = ItemKNNCFRecommender(URM_train=data.get_urm_train())
# rec.fit()
# recs = rec.recommend_batch(userids=data.get_target_playlists(), type='ITEM')
# rec.evaluate(recs, test_urm=data.get_urm_test())
