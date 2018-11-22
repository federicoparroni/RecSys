import numpy as np
from recommenders.recommender_base import RecommenderBase
import data.data as data as d
import pyximport
import time
import sys
import utils.log as log
from inout import importexport
pyximport.install(setup_args={"script_args":[],
                              "include_dirs":np.get_include()},
                  reload_support=True)

class SLIM_BPR(RecommenderBase):

    """
    Learns a items similarity matrix W. The estimated URM (URM^) can be obtained by URM*W.
    In order to learn it, BPR loss is used.
    Various optimization methods are available. Worth mentioning are 'adagrad' and 'sgd'.
    """

    def __init__(self, URM_train):
        self.URM_train=URM_train
        self.n_users = URM_train.shape[0]
        self.n_items = URM_train.shape[1]

    def fit(self, epochs=30, URM_test=None, user_ids=None, batch_size = 1000, validate_every_N_epochs = 1,
            start_validation_after_N_epochs = 0, lambda_i = 0.0, lambda_j = 0.0,
            learning_rate = 0.01, topK = 200, sgd_mode='adagrad'):

        """
        train the model finding matrix W

        Parameters
        ----------
        :param (int) epochs
        :param (csr_matrix) URM_test: needed if we'd like to perform validation
        :param (list) user_ids: needed if we'd like to perform validation
        :param (int) batch_size: after how many items the params should be updated
        :param (int) validate_every_N_epochs: how often the MAP evaluation should be displayed
        :param (int) start_validation_after_N_epochs
        :param (float) lambda_i: first regularization term
        :param (float) lambda_j: second regularization term
        :param (float) learning_rate: algorithm learning rate
        :param (int) topK: how many elements should be taken into account while computing URM*W
        :param (string) sgd_mode: optimization algorithm

        Returns
        -------

        """

        self.sgd_mode = sgd_mode

        from Cython.SLIM_BPR.SLIM_BPR_Cython_Epoch import SLIM_BPR_Cython_Epoch

        self.cythonEpoch = SLIM_BPR_Cython_Epoch(self.URM_train,
                                                 sparse_weights = False,
                                                 topK=topK,
                                                 learning_rate=learning_rate,
                                                 li_reg = lambda_i,
                                                 lj_reg = lambda_j,
                                                 batch_size=1,
                                                 symmetric = True,
                                                 sgd_mode = sgd_mode)


        # Cal super.fit to start training
        self._fit_alreadyInitialized(epochs=epochs,
                                    logFile=None,
                                    URM_test=URM_test,
                                    user_ids=user_ids,
                                    filterTopPop=False,
                                    minRatingsPerUser=1,
                                    batch_size=batch_size,
                                    validate_every_N_epochs=validate_every_N_epochs,
                                    start_validation_after_N_epochs=start_validation_after_N_epochs,
                                    lambda_i = lambda_i,
                                    lambda_j = lambda_j,
                                    learning_rate = learning_rate,
                                    topK = topK)

    def _fit_alreadyInitialized(self, epochs=30, logFile=None, URM_test=None, user_ids = None, filterTopPop = False, minRatingsPerUser=1,
                                batch_size = 1000, validate_every_N_epochs = 1, start_validation_after_N_epochs = 0,
                                lambda_i = 0.0025, lambda_j = 0.00025, learning_rate = 0.05, topK = False):
        if(topK != False and topK<1):
            raise ValueError("TopK not valid. Acceptable values are either False or a positive integer value. Provided value was '{}'".format(topK))

        self.topK = topK
        self.batch_size = batch_size
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j
        self.learning_rate = learning_rate
        start_time_train = time.time()

        for currentEpoch in range(epochs):

            start_time_epoch = time.time()
            if self.batch_size>0:
                self._epochIteration()
            else:
                print("No batch not available")

            if ((currentEpoch +1 )% validate_every_N_epochs == 0) and currentEpoch >= start_validation_after_N_epochs:
                if URM_test == None or user_ids == None:
                    raise ValueError("Validation cannot be performed without URM_test and user_ids!")

                print("Evaluation begins")
                self._updateSimilarityMatrix()
                recs = self.recommend_batch(user_ids)
                self.evaluate(recs, URM_test)

            print("Epoch {} of {} complete in {:.2f} minutes".format(currentEpoch+1, epochs,
                                                                     float(time.time() - start_time_epoch) / 60))

        self._updateSimilarityMatrix()
        print("Fit completed in {:.2f} minutes".format(float(time.time() - start_time_train) / 60))
        sys.stdout.flush()

    def _updateSimilarityMatrix(self):
        self.S = self.cythonEpoch.get_S()
        self.W_sparse = self.S

    def _epochIteration(self):
        self.cythonEpoch.epochIteration_Cython()

    def recommend_batch(self, userids, N=10, urm=None, filter_already_liked=True, with_scores=False,
                        items_to_exclude=[], verbose=False):
        user_profile_batch = self.URM_train[userids]
        scores_array = user_profile_batch.dot(self.W_sparse).toarray()

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
            n = self.URM_train.shape[1] - 1
        else:
            n = N

        # compute the scores using the dot product
        user_profile = self.URM_train[userid]
        scores = user_profile.dot(self.W_sparse).toarray().ravel()

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


    def _filter_seen_on_scores(self, user_id, scores):

        seen = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]

        scores[seen] = -np.inf
        return scores

    def get_r_hat(self):
        pass

# test
# s = SLIM_BPR(d.get_urm())
# s.fit(epochs=100, validate_every_N_epochs=101, learning_rate=1e-2,
#       lambda_i = 1e-4, lambda_j = 1e-4)
# recs = s.recommend_batch(d.get_target_playlists(), urm=d.get_urm_train(), N=10, filter_already_liked=True, with_scores=False)
# # s.evaluate(recs, d.get_urm_test(), print_result=True)
# importexport.exportcsv(recs, 'submission', 'SLIM_BPR')
