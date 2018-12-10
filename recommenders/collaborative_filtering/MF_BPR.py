import numpy as np
import time
from tqdm import tqdm
import scipy as sc
from scipy import sparse
from recommenders.recommender_base import RecommenderBase
import utils.log as log

"""
Matrix Factorization with a Bayesian Personalized Ranking approach.
taken from https://github.com/tmscarla/recsys-toolbox
"""

class MFBPR(RecommenderBase):

    def sampleTriplet(self):
        user_id = np.random.choice(self.n_users)

        # Get user seen items and choose one
        userSeenItems = self.URM[user_id, :].indices
        pos_item_id = np.random.choice(userSeenItems)

        negItemSelected = False
        while (not negItemSelected):
            neg_item_id = np.random.randint(0, self.n_items)

            if (neg_item_id not in userSeenItems):
                negItemSelected = True

        return user_id, pos_item_id, neg_item_id

    def epochIteration(self):
        numPositiveIteractions = int(self.URM.nnz * 0.1)

        # Uniform user sampling without replacement
        for it in tqdm(range(numPositiveIteractions)):
            u, i, j = self.sampleTriplet()
            self.update_factors(u, i, j)

    def update_factors(self, u, i, j, update_u=True, update_i=True, update_j=True):
        # SGD update
        x = np.dot(self.user_factors[u, :], self.item_factors[i, :] - self.item_factors[j, :])

        z = 1.0 / (1.0 + np.exp(x))

        if update_u:
            d = (self.item_factors[i, :] - self.item_factors[j, :]) * z \
                - self.user_regularization * self.user_factors[u, :]
            self.user_factors[u, :] += self.learning_rate * d
        if update_i:
            d = self.user_factors[u, :] * z - self.positive_item_regularization * self.item_factors[i, :]
            self.item_factors[i, :] += self.learning_rate * d
        if update_j:
            d = -self.user_factors[u, :] * z - self.negative_item_regularization * self.item_factors[j, :]
            self.item_factors[j, :] += self.learning_rate * d

    def fit(self, URM, n_factors=10, learning_rate=0.1, epochs=10, user_regularization=0.01, positive_item_regularization=0.01, negative_item_regularization=0.01):
        self.URM = URM
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.user_regularization = user_regularization
        self.positive_item_regularization = positive_item_regularization
        self.negative_item_regularization = negative_item_regularization

        self.n_users = self.URM.shape[0]
        self.n_items = self.URM.shape[1]
        self.user_factors = np.random.random_sample((self.n_users, n_factors))
        self.item_factors = np.random.random_sample((self.n_items, n_factors))

        print('Fitting MFBPR...')

        for numEpoch in range(self.epochs):
            print('Epoch:', numEpoch)
            self.epochIteration()

    def run(self):
        pass

    def get_r_hat(self):
        pass

    def _filter_seen_on_scores(self, user_id, scores):
        seen = self.URM.indices[self.URM.indptr[user_id]:self.URM.indptr[user_id + 1]]
        scores[seen] = -np.inf
        return scores

    def recommend_batch(self, userids, N=10, urm=None, filter_already_liked=True, with_scores=False,
                        items_to_exclude=[], verbose=False):
        user_profile_batch = self.URM[userids]
        scores_array = np.dot(self.user_factors[userids], self.item_factors.T)

        if filter_already_liked:
            scores_array[user_profile_batch.nonzero()] = -np.inf

        if len(items_to_exclude) > 0:
            raise NotImplementedError('Items to exclude functionality is not implemented yet')

        i = 0
        l = []
        for row_index in range(scores_array.shape[0]):
            scores = scores_array[row_index]

            relevant_items_partition = (-scores).argpartition(N)[0:N]
            relevant_items_partition_sorting = np.argsort(-scores[relevant_items_partition])
            ranking = relevant_items_partition[relevant_items_partition_sorting]
            if with_scores:
                s = scores_array[row_index, ranking]
                l.append([userids[row_index]] + [list(zip(list(ranking), list(s)))])
            else:
                l.append([userids[row_index]] + list(ranking))
            if verbose:
                 i += 1
                 log.progressbar(i, scores_array.shape[0], prefix='Building recommendations ')
        return l

    def recommend(self, userid, N=10, urm=None, filter_already_liked=True, with_scores=False, items_to_exclude=[]):
        if N == None:
            n = self.URM.shape[1] - 1
        else:
            n = N
        scores = np.dot(self.user_factors[userid], self.item_factors.T)

        if filter_already_liked:
            scores = self._filter_seen_on_scores(userid, scores)

        if len(items_to_exclude) > 0:
            raise NotImplementedError('Items to exclude functionality is not implemented yet')

        relevant_items_partition = (-scores).argpartition(n)[0:n]
        relevant_items_partition_sorting = np.argsort(-scores[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]

        if with_scores:
            best_scores = scores[ranking]
            return [userid] + [list(zip(list(ranking), list(best_scores)))]
        else:
            return [userid] + list(ranking)

import data.data as d
r = MFBPR()
r.fit(d.get_urm_train_1(), epochs=10, n_factors=10)
recs = r.recommend_batch(userids=d.get_target_playlists())
r.evaluate(recs, d.get_urm_test_1())