import numpy as np
import time
import math
import scipy as sc
import utils.log as log
import data.data as d
from tqdm import tqdm
from scipy import sparse
from recommenders.recommender_base import RecommenderBase
from cythoncompiled.MF_BPR.MF_BPR import MFBPR_Epoch

"""
Matrix Factorization with a Bayesian Personalized Ranking approach.
taken from https://github.com/tmscarla/recsys-toolbox
"""

class MFBPR(RecommenderBase):

    def fit(self, URM, n_factors=10, learning_rate=1e-4, epochs=10, user_regularization=0.001, positive_item_regularization=0.001, negative_item_regularization=0.001, evaluate_every=1):
        self.URM = URM
        self.epochs = epochs
        self.n_users = self.URM.shape[0]
        self.n_items = self.URM.shape[1]

        e = MFBPR_Epoch(URM, n_factors=n_factors, learning_rate=learning_rate, user_regularization=user_regularization, 
                        positive_item_regularization=positive_item_regularization, negative_item_regularization=negative_item_regularization)
        print('Fitting MFBPR...')

        for numEpoch in range(self.epochs):
            print('Epoch:', numEpoch)
            e.epochIteration()
            if (numEpoch + 1) % evaluate_every == 0:
                self.user_factors, self.item_factors = e.get_user_item_factors()
                recs = self.recommend_batch(userids=d.get_target_playlists())
                self.evaluate(recs, d.get_urm_test_1())

        self.user_factors, self.item_factors = e.get_user_item_factors()

        # let's see how fine it performs in the test set:
        # getting as positive sample a semple in the test set but not in the training
        trials = 10000
        count_wrong = 0
        for _ in range(trials):
            test = d.get_urm_test_1()
            user_id = np.random.choice(self.n_users)
            user_seen_items = d.get_urm()[user_id, :].indices
            test_items = test[user_id, :].indices
            pos_item_id = np.random.choice(test_items)
            neg_item_selected = False
            while (not neg_item_selected):
                neg_item_id = np.random.randint(0, self.n_items)
                if (neg_item_id not in user_seen_items):
                    neg_item_selected = True
            xui = np.dot(self.user_factors[user_id, :], self.item_factors[pos_item_id, :])
            xuj = np.dot(self.user_factors[user_id, :], self.item_factors[neg_item_id, :])
            xuij = xui - xuj
            if xuij < 0:
                count_wrong += 1
            # print('u: {}, i: {}, j: {}. xui - xuj: {}'.format(user_id, pos_item_id, neg_item_id, xuij))
        print('percentange of wrong preferences in test set: {}'.format(count_wrong/trials))

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
r.fit(d.get_urm_train_1(),
      epochs=30,
      n_factors=150,
      learning_rate=1e-2,
      user_regularization=1e-2,
      positive_item_regularization=1e-2,
      negative_item_regularization=1e-2,
      evaluate_every=1)
