
import utils.check_matrix_format as cm
from recommenders.SimilarityMatrixRecommender import SimilarityMatrixRecommender
from recommenders.Similarity.Compute_Similarity import Compute_Similarity
import data.data as data
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

    def recommend_batch(self, userids, N=10, filter_already_liked=True, verbose=False):
        """
        look for comment on superclass method
        """
        # compute the scores using the dot product
        user_profile_batch = self.URM_train[userids]

        scores_array = np.dot(self.W_sparse[userids], self.URM_train)

        """
        To exclude already_liked items perform a boolean indexing and replace their score with -inf
        Seen items will be at the bottom of the list but there is no guarantee they'll NOT be
        recommended
        """
        if filter_already_liked:
            scores_array[user_profile_batch.nonzero()] = -np.inf

        # magic code 🔮 to take the top N recommendations
        ranking = np.zeros((scores_array.shape[0], N), dtype=np.int)
        for row_index in range(scores_array.shape[0]):
            scores = scores_array[row_index]
            scores = scores.todense()
            relevant_items_partition = (-scores).argpartition(N)[0, 0:N]
            relevant_items_partition_sorting = np.argsort(-scores[0, relevant_items_partition])
            ranking[row_index] = relevant_items_partition[0, relevant_items_partition_sorting[0, 0:N]]

        """
        add target id in a way that recommendations is a list as follows
        [ [playlist1_id, id1, id2, ....., id10], ...., [playlist_id2, id1, id2, ...] ]
        """
        np_target_id = np.array(userids)
        target_id_t = np.reshape(np_target_id, (len(np_target_id), 1))
        recommendations = np.concatenate((target_id_t, ranking), axis=1)

        return recommendations

def evaluate(recommendations, test_urm, at_k=10, verbose=True):
    """
    Return the MAP@k evaluation for the provided recommendations
    computed with respect to the test_urm

    Parameters
    ----------
    recommendations : list
        List of recommendations, where a recommendation
        is a list (of length N+1) of playlist_id and N items_id:
            [   [7,   18,11,76, ...] ,
                [13,  65,83,32, ...] ,
                [25,  30,49,65, ...] , ... ]
    test_urm : csr_matrix
        A sparse matrix
    at_k : int, optional
        The number of items to compute the precision at

    Returns
    -------
    MAP@k: (float) MAP for the provided recommendations
    """
    if not at_k > 0:
        log.error('Invalid value of k {}'.format(at_k))
        return

    aps = 0.0
    for r in recommendations:
        row = test_urm.getrow(r[0]).indices
        m = min(at_k, len(row))

        ap = 0.0
        n_elems_found = 0.0
        for j in range(1, m + 1):
            if r[j] in row:
                n_elems_found += 1
                ap = ap + n_elems_found / j
        if m > 0:
            ap = ap / m
            aps = aps + ap

    result = aps / len(recommendations)
    if verbose:
        log.warning('MAP: {}'.format(result))
    return result

rec = UserKNNCFRecommender(URM_train=data.get_urm())
rec.fit()
sps.save_npz('raw_data/saved_r_hat/userKNN', np.dot(rec.W_sparse[data.get_target_playlists()], rec.URM_train))
#recs = rec.recommend_batch(userids=data.get_target_playlists())
#evaluate(recommendations=recs, test_urm=data.get_urm_test())
