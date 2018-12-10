import numpy as np
import time
import math
import scipy as sc
import utils.log as log
import data.data as d
from tqdm import tqdm
from recommenders.recommender_base import RecommenderBase

cdef class fit_MFBPR:

    cdef int[:] URM_indices, URM_indptr, URM_data
    cdef int n_factors
    cdef double learning_rate
    cdef int epochs
    cdef double user_regularization
    cdef double positive_item_regularization
    cdef double negative_item_regularization
    cdef int n_users, n_items
    cdef double[:,:] user_factors, item_factors
    cdef double loss

    def sampleTriplet(self):
        cdef int user_id
        cdef int[:] user_seen_items
        cdef int pos_item_id
        cdef int neg_item_selected

        user_id = np.random.choice(self.n_users)

        # Get user seen items and choose one
        user_seen_items = self.URM_data[self.URM_indptr[user_id] : self.URM_indptr[user_id + 1]]
        pos_item_id = np.random.choice(user_seen_items)

        neg_item_selected = False
        while (not neg_item_selected):
            neg_item_id = np.random.randint(0, self.n_items)

            if (neg_item_id not in user_seen_items):
                neg_item_selected = True

        return user_id, pos_item_id, neg_item_id

    def epochIteration(self):
        cdef int num_positive_interactions
        cdef int u, i, j
        
        # num_positive_interactions = int(len(self.URM_data) * 0.1)
        # num_positive_interactions = 1211791
        num_positive_interactions = 100000

        # self.loss = 0
        # self.user_list = []
        # self.countwrong = 0

        # Uniform user sampling without replacement
        for _ in tqdm(range(num_positive_interactions)):
            u, i, j = self.sampleTriplet()
            self.update_factors(u, i, j)
        
        print('loss: {}'.format(self.loss / num_positive_interactions))
        # print('perc of wrong bpr estimates: {}'.format(self.countwrong / num_positive_interactions))

    def update_factors(self, u, i, j):
        # SGD update
        cdef double xui, xuj, xuij
        cdef double loss
        cdef double z
        cdef double[:] user_factor_c, item_factors_c_i, item_factors_c_j

        xui = np.dot(self.user_factors[u, :], self.item_factors[i, :])
        xuj = np.dot(self.user_factors[u, :], self.item_factors[j, :])
        xuij = xui - xuj
        self.loss += xuij
        z = np.exp(-xuij)/(1 + np.exp(-xuij))
        
        # if xuij < 0:
        #   self.countwrong += 1

        user_factor_c = self.user_factors[u, :].copy()
        item_factors_c_i = self.item_factors[i, :].copy()
        item_factors_c_j = self.item_factors[j, :].copy()

        cdef double[:] new_user_factor
        new_user_factor = user_factor_c + self.learning_rate * (
           z * (np.subtract(item_factors_c_i, item_factors_c_j)) + np.multiply(self.user_regularization, user_factor_c))
        self.user_factors[u, :] = new_user_factor

        cdef double[:] new_item_factor_i
        new_item_factor_i = self.item_factors[i, :] + self.learning_rate * (
            np.multiply(z, user_factor_c) + np.multiply(self.positive_item_regularization, item_factors_c_i))
        self.item_factors[i, :] = new_item_factor_i

        cdef double[:] new_item_factor_j
        new_item_factor_j = self.item_factors[j, :] + self.learning_rate * (
            z * (np.negative(user_factor_c)) + np.multiply(self.negative_item_regularization, item_factors_c_j))
        self.item_factors[j, :] = new_item_factor_j

    def fit(self, URM, n_factors=10, learning_rate=1e-4, epochs=10, user_regularization=0.001, positive_item_regularization=0.001, 
            negative_item_regularization=0.001, evaluate_every=1):
        
        self.URM_indices = URM.indices
        self.URM_data = URM.data.astype(np.int32)
        self.URM_indptr = URM.indptr 
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.user_regularization = user_regularization
        self.positive_item_regularization = positive_item_regularization
        self.negative_item_regularization = negative_item_regularization

        self.n_users = URM.shape[0]
        self.n_items = URM.shape[1]
        self.user_factors = np.random.normal(0, math.sqrt(self.user_regularization), size=(self.n_users, n_factors))
        self.item_factors = np.random.normal(0, math.sqrt(self.user_regularization), size=(self.n_items, n_factors))

        print('Fitting MFBPR...')

        for numEpoch in range(self.epochs):
            print('Epoch:', numEpoch)
            self.epochIteration()
            # if (numEpoch + 1) % evaluate_every == 0:
            #     recs = r.recommend_batch(userids=d.get_target_playlists())
            #     r.evaluate(recs, d.get_urm_test())
