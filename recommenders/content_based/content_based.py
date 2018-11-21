"""
Content based recommender.
"""

from recommenders.distance_based_recommender import DistanceBasedRecommender
import utils.log as log
import numpy as np
import similaripy as sim
import data

class ContentBasedRecommender(DistanceBasedRecommender):
    """
    Computes the recommendations for a user by looking for the most similar items that he
    has already interacted with.
    """

    def __init__(self):
        super(ContentBasedRecommender, self).__init__()

    def fit(self, icm, k, distance, shrink=0, threshold=0, alpha=None, beta=None, l=None, c=None):
        """
        Initialize the model and compute the similarity matrix S with a distance metric.
        Access the similarity matrix using: self._sim_matrix

        Parameters
        ----------
        icm : csr_matrix
            The ICM matrix of shape (number_items, number_features).
        k : int
            K nearest neighbour to consider.
        distance : str
            One of the supported distance metrics, check collaborative_filtering_base constants.
        shrink : float, optional
            Shrink term used in the normalization
        alpha: float, optional, included in [0,1]
        beta: float, optional, included in [0,1]
        l: float, optional, balance coefficient used in s_plus distance, included in [0,1]
        c: float, optional, cosine coefficient, included in [0,1]
        """
        super(ContentBasedRecommender, self).fit(matrix=icm.T, k=k, distance=distance, shrink=shrink, threshold=threshold, implicit=False, alpha=alpha, beta=beta, l=l, c=c)
    
    def recommend(self, userid, urm, N=10, filter_already_liked=True, with_scores=False, items_to_exclude=[]):
        if not userid >= 0:
            log.error('Invalid user id')
            return None
        return self.recommend_batch([userid], urm, N, filter_already_liked, with_scores, items_to_exclude)

    def recommend_batch(self, userids, urm, N=10, filter_already_liked=True, with_scores=False, items_to_exclude=[], verbose=False):
        if not self._has_fit():
            return None

        if userids is not None:
            if len(userids) > 0:
                matrix = urm[userids] if urm is not None else self._matrix[userids]
            else:
                return []
        else:
            print('Recommending for all users...')
            matrix = urm if urm is not None else self._matrix

        # compute the R^ by multiplying R•S
        r_hat = sim.dot_product(matrix, self._sim_matrix, target_rows=None, k=data.N_TRACKS, format_output='csr', verbose=verbose)
        
        if filter_already_liked:
            user_profile_batch = matrix
            r_hat[user_profile_batch.nonzero()] = -np.inf
        if len(items_to_exclude)>0:
            # TO-DO: test this part because it does not work!
            r_hat = r_hat.T
            r_hat[items_to_exclude] = -np.inf
            r_hat = r_hat.T
        
        recommendations = self._extract_top_items(r_hat, N=N)
        return self._insert_userids_as_first_col(userids, recommendations).tolist()
