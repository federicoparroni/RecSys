from sklearn.utils.extmath import randomized_svd
from recommenders.recommender_base import RecommenderBase
import scipy.sparse as sps
import numpy as np


class Pure_SVD(RecommenderBase):

    def __init__(self, urm):
        self.urm = urm

    def fit(self, num_factors=50):
        U, Sigma, VT = randomized_svd(self.urm,
                                      n_components=num_factors,
                                      random_state=None)
        self.s_Vt = sps.diags(Sigma)*VT
        self.U = U


    def get_r_hat(self, load_from_file=False, path=''):
        """
        :param load_from_file: if the matrix has been saved can be set to true for load it from it
        :param path: path in which the matrix has been saved
        -------
        :return the extimated urm from the recommender
        """
        pass

    def recommend(self, userid, N=10, urm=None, filter_already_liked=True, with_scores=False, items_to_exclude=[]):
        """
        Recommend the N best items for the specified user

        Parameters
        ----------
        userid : int
            The user id to calculate recommendations for
        urm : csr_matrix
            A sparse matrix of shape (number_users, number_items). This allows to look
            up the liked items and their weights for the user. It is used to filter out
            items that have already been liked from the output, and to also potentially
            giving more information to choose the best items for this user.
        N : int, optional
            The number of recommendations to return
        items_to_exclude : list of ints, optional
            List of extra item ids to filter out from the output

        Returns
        -------
        list
            List of length N of (itemid, score) tuples: [ (18,0.7), (51,0.5), ... ]
        """
        pass

    def recommend_batch(self, userids,  N=10, filter_already_liked=True, with_scores=False, items_to_exclude=[]):

        user_profile = self.urm[userids]

        U_filtered = self.U[userids]
        scores = U_filtered.dot(self.s_Vt)

        if filter_already_liked:
            scores[user_profile.nonzero()] = -np.inf

        ranking = np.zeros((scores.shape[0], N), dtype=np.int)
        #scores = scores.todense()

        for row_index in range(scores.shape[0]):
            scores_row = scores[row_index]

            relevant_items_partition = (-scores_row).argpartition(N)[0:N]
            relevant_items_partition_sorting = np.argsort(-scores_row[relevant_items_partition])
            ranking[row_index] = relevant_items_partition[relevant_items_partition_sorting[0:N]]

        recommendations = self._insert_userids_as_first_col(userids, ranking)
        return recommendations