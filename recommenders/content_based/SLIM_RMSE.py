from recommenders.recommender_base import RecommenderBase
from utils import check_matrix_format as cm
import numpy as np
from sklearn.linear_model import ElasticNet
import time
import scipy.sparse as sps
import sys
import multiprocessing
from multiprocessing import Pool
from functools import partial
import pathos.pools as pp


import numpy as np
import scipy.sparse as sps

from sklearn.linear_model import ElasticNet

import time, sys


class SLIMElasticNetRecommender(RecommenderBase):
    """
    Train a Sparse Linear Methods (SLIM) item similarity model.
    NOTE: ElasticNet solver is parallel, a single intance of SLIM_ElasticNet will
          make use of half the cores available

    See:
        Efficient Top-N Recommendation by Linear Regression,
        M. Levy and K. Jack, LSRS workshop at RecSys 2013.
        https://www.slideshare.net/MarkLevy/efficient-slides

        SLIM: Sparse linear methods for top-n recommender systems,
        X. Ning and G. Karypis, ICDM 2011.
        http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf
    """

    RECOMMENDER_NAME = "SLIMElasticNetRecommender"

    def __init__(self, URM_train):
        self.URM_train = URM_train

    def _partial_fit(self, URM_train, currentItem):

        if self.l1_penalty + self.l2_penalty != 0:
            self.l1_ratio = self.l1_penalty / (self.l1_penalty + self.l2_penalty)
        else:
            print("SLIM_ElasticNet: l1_penalty+l2_penalty cannot be equal to zero, setting the ratio l1/(l1+l2) to 1.0")
            self.l1_ratio = 1.0

        # initialize the ElasticNet model
        model = ElasticNet(alpha=1.0,
                            l1_ratio=self.l1_ratio,
                            positive=self.positive_only,
                            fit_intercept=False,
                            copy_X=False,
                            precompute=True,
                            selection='random',
                            max_iter=5,
                            tol=1e-4)

        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0

        start_time = time.time()
        start_time_printBatch = start_time

        #print(currentItem)
        print(str(self.count*100/20635)+'%')
        self.count += 1



        # get the target column
        y = URM_train[:, currentItem].toarray()

        # set the j-th column of X to zero
        start_pos = URM_train.indptr[currentItem]
        end_pos = URM_train.indptr[currentItem + 1]

        current_item_data_backup = URM_train.data[start_pos: end_pos].copy()
        URM_train.data[start_pos: end_pos] = 0.0

        # fit one ElasticNet model per column
        model.fit(self.URM_train, y)

        # self.model.coef_ contains the coefficient of the ElasticNet model
        # let's keep only the non-zero values

        # Select topK values
        # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
        # - Partition the data to extract the set of relevant items
        # - Sort only the relevant items
        # - Get the original item index

        nonzero_model_coef_index = model.sparse_coef_.indices
        nonzero_model_coef_value = model.sparse_coef_.data

        local_topK = min(len(nonzero_model_coef_value) - 1, self.topK)

        relevant_items_partition = (-nonzero_model_coef_value).argpartition(local_topK)[0:local_topK]
        relevant_items_partition_sorting = np.argsort(-nonzero_model_coef_value[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]

        values = model.coef_[ranking]
        rows = ranking
        cols = [currentItem] * len(ranking)

        return values, rows, cols

    def fit(self, l1_penalty=1, l2_penalty=10, positive_only=True, topK=100, workers=multiprocessing.cpu_count()):
        self.count = 0
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.positive_only = positive_only
        self.l1_ratio = self.l1_penalty / (self.l1_penalty + self.l2_penalty)
        self.topK = topK

        self.workers = workers
        self.URM_train = sps.csc_matrix(self.URM_train)
        n_items = self.URM_train.shape[1]

        # oggetto riferito alla funzione nel quale predefinisco parte dell'input
        _pfit = partial(self._partial_fit, self.URM_train)

        # creo un pool con un certo numero di processi
        pool = pp.ProcessPool(self.workers)

        # avvio il pool passando la funzione (con la parte fissa dell'input)
        # e il rimanente parametro, variabile
        res = pool.map(_pfit, np.arange(n_items))

        # res contains a vector of (values, rows, cols) tuples
        values, rows, cols = [], [], []
        for values_, rows_, cols_ in res:
            values.extend(values_)
            rows.extend(rows_)
            cols.extend(cols_)

        # generate the sparse weight matrix
        self.W_sparse = sps.csc_matrix((values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)

    def recommend_batch(self, userids, N=10, filter_already_liked=True, with_scores=False, items_to_exclude=[], verbose=False):
        # compute the scores using the dot product
        user_profile = self.URM_train[userids]
        scores = user_profile.dot(self.W_sparse)

        if filter_already_liked:
            scores[user_profile.nonzero()] = -np.inf

        ranking = np.zeros((scores.shape[0], N), dtype=np.int)
        scores = scores.todense()

        for row_index in range(scores.shape[0]):
            scores_row = scores[row_index]

            relevant_items_partition = (-scores_row).argpartition(N)[0, 0:N]
            relevant_items_partition_sorting = np.argsort(-scores_row[0, relevant_items_partition])
            ranking[row_index] = relevant_items_partition[0, relevant_items_partition_sorting[0, 0:N]]

        """
        add target id in a way that recommendations is a list as follows
        [ [playlist1_id, id1, id2, ....., id10], ...., [playlist_id2, id1, id2, ...] ]
        """
        np_target_id = np.array(userids)
        target_id_t = np.reshape(np_target_id, (len(np_target_id), 1))
        recommendations = np.concatenate((target_id_t, ranking), axis=1)

        return recommendations




"""
class SLIM_RMSE(RecommenderBase):


    def __init__(self, URM):
        self.urm = URM

    def fit(self, l1_penalty=1, l2_penalty=10, positive_only=False, topK = 100):

        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.positive_only = positive_only
        self.l1_ratio = self.l1_penalty / (self.l1_penalty + self.l2_penalty)
        self.topK = topK

        X = cm.check_matrix(self.urm, 'csc', dtype=np.float32)

        n_items = X.shape[1]

        # initialize the ElasticNet model
        self.model = ElasticNet(alpha=1.0,
                                l1_ratio=self.l1_ratio,
                                positive=self.positive_only,
                                fit_intercept=False,
                                copy_X=False,
                                precompute=True,
                                selection='random',
                                max_iter=1,
                                tol=1e-4)

        # we'll store the W matrix into a sparse csr_matrix
        # let's initialize the vectors used by the sparse.csc_matrix constructor
        values, rows, cols = [], [], []
        start_time = time.time()
        start_time_printBatch = start_time

        # fit each item's factors sequentially (not in parallel)
        for currentItem in range(n_items):
            print(str(currentItem*100/n_items) + "%")
            # get the target column
            y = X[:, currentItem].toarray()
            # set the j-th column of X to zero
            startptr = X.indptr[currentItem]
            endptr = X.indptr[currentItem + 1]
            bak = X.data[startptr: endptr].copy()
            X.data[startptr: endptr] = 0.0
            # fit one ElasticNet model per column
            self.model.fit(X, y)

            print(self.model.coef_.max())
            print(self.model.coef_.min())

            # self.model.coef_ contains the coefficient of the ElasticNet model
            # let's keep only the non-zero values
            #nnz_idx = self.model.coef_ > 0.0

            # Select topK values
            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index
            relevant_items_partition = (-self.model.coef_).argpartition(self.topK)[0:self.topK]
            relevant_items_partition_sorting = np.argsort(-self.model.coef_[relevant_items_partition])
            ranking = relevant_items_partition[relevant_items_partition_sorting]

            notZerosMask = self.model.coef_[ranking] > 0.0
            ranking = ranking[notZerosMask]

            values.extend(self.model.coef_[ranking])
            rows.extend(ranking)
            cols.extend([currentItem]*len(ranking))

            # finally, replace the original values of the j-th column
            X.data[startptr:endptr] = bak


            if time.time() - start_time_printBatch > 300:
                print("Processed {} ( {:.2f}% ) in {:.2f} minutes. Columns per second: {:.0f}".format(
                                  currentItem,
                                  100.0* float(currentItem)/n_items,
                                  (time.time()-start_time)/60,
                                  float(currentItem)/(time.time()-start_time)))
                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()


        # generate the sparse weight matrix
        self.W_sparse = sps.csc_matrix((values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)

    def recommend_batch(self, userids, N=10, with_scores=True, filter_already_liked=True, items_to_exclude=[], verbose=False):
        #TODO: IMPLEMENT item_to_exclude
        #TODO: IMPLEMENT with_scores

        # compute the scores using the dot product
        user_profile_batch = self.urm[userids]

        scores_array = np.array(user_profile_batch.dot(self.W_sparse))

        # To exclude seen items perform a boolean indexing and replace their score with -inf
        # Seen items will be at the bottom of the list but there is no guarantee they'll NOT be
        # recommended
        if filter_already_liked:
            scores_array[user_profile_batch.nonzero()] = -np.inf

        # rank items and mirror column to obtain a ranking in descending score
        # ranking = (-scores_array).argsort(axis=1)
        # ranking = np.fliplr(ranking)
        # ranking = ranking[:,0:n]

        ranking = np.zeros((scores_array.shape[0], N), dtype=np.int)

        for row_index in range(scores_array.shape[0]):
            scores = scores_array[row_index]

            relevant_items_partition = (-scores).argpartition(N)[0:N]
            relevant_items_partition_sorting = np.argsort(-scores[relevant_items_partition])
            ranking[row_index] = relevant_items_partition[relevant_items_partition_sorting]


        np_target_id = np.array(userids)
        target_id_t = np.reshape(np_target_id, (len(np_target_id), 1))
        recommendations = np.concatenate((target_id_t, ranking), axis=1)

        return recommendations

    def get_sim_matrix(self):
        return self.W_sparse
"""