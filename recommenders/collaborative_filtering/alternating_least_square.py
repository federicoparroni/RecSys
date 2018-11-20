import numpy as np
import implicit # The Cython library
from recommenders.recommender_base import RecommenderBase


class AlternatingLeastSquare(RecommenderBase):

    """
    Reference: http://yifanhu.net/PUB/collaborative_filtering.pdf (PAPER)
               https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe (SIMPLE EXPLANATION)

    Implementation of Alternating Least Squares with implicit data. We iteratively
    compute the user (x_u) and item (y_i) vectors using the following formulas:

    x_u = ((Y.T*Y + Y.T*(Cu - I) * Y) + lambda*I)^-1 * (X.T * Cu * p(u))
    y_i = ((X.T*X + X.T*(Ci - I) * X) + lambda*I)^-1 * (Y.T * Ci * p(i))

    [link text](http://www.example.com)
    """
    def __init__(self, URM):
        self.urm = URM


    def fit(self, factors=100, regularization=0.01, iterations=100, alpha_val=25):
        """
        train the model finding the two matrices U and V: U*V.T=R  (R is the extimated URM)

        Parameters
        ----------
        :param (int) factors: How many latent features we want to compute.
        :param (float) regularization: lambda_val regularization value
        :param (int) iterations: How many times we alternate between fixing and updating our user and item vectors
        :param (int) alpha_val: The rate in which we'll increase our confidence in a preference with more interactions.

        Returns
        -------
        :return (csr_matrix) user_vecs: matrix N_user x factors
        :return (csr_matrix) item_vecs: matrix N_item x factors

        """
        sparse_item_user = self.urm.T

        # Initialize the als model and fit it using the sparse item-user matrix
        self._model = implicit.als.AlternatingLeastSquares(factors=factors,
                                                          regularization=regularization,
                                                          iterations=iterations)

        # Calculate the confidence by multiplying it by our alpha value.
        data_conf = (sparse_item_user*alpha_val).astype('double')

        # Fit the model
        self._model.fit(data_conf)

        # set the user and item vectors for our model R = user_vecs * item_vecs.T
        self.user_vecs = self._model.user_factors
        self.item_vecs = self._model.item_factors


    def recommend(self, userid, N=10, urm=None, filter_already_liked=True, with_scores=True, items_to_exclude=[]):
        """
        look for comment on superclass method
        """
        # compute the scores using the dot product
        user_profile = self.urm[userid]

        scores = np.dot(self.user_vecs[userid], self.item_vecs.T)

        # To exclude seen items perform a boolean indexing and replace their score with -inf
        # Seen items will be at the bottom of the list but there is no guarantee they'll NOT be
        # recommended
        if filter_already_liked:
            scores[user_profile.nonzero()[1]] = -np.inf

        relevant_items_partition = np.argpartition((-scores), N)[0:N]
        relevant_items_partition_sorting = np.argsort(-scores[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]

        return ranking

    def recommend_batch(self, userids, N=10, filter_already_liked=True, with_scores=False, items_to_exclude=[],
                        verbose=False):
        """
        look for comment on superclass method
        """
        # compute the scores using the dot product
        user_profile_batch = self.urm[userids]

        scores_array = np.dot(self.user_vecs[userids], self.item_vecs.T)

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
            relevant_items_partition = (-scores).argpartition(N)[0:N]
            relevant_items_partition_sorting = np.argsort(-scores[relevant_items_partition])
            ranking[row_index] = relevant_items_partition[relevant_items_partition_sorting]
        
        # include userids as first column
        recommendations = self._insert_userids_as_first_col(userids, ranking)

        return recommendations


