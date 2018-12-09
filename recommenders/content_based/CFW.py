#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/09/17

@author: Maurizio Ferrari Dacrema
"""

from recommenders.recommender_base import RecommenderBase
from scipy.sparse import linalg
from scipy.sparse import csr_matrix
import time, sys
import numpy as np
import pyximport
import similaripy as sim
from recommenders.collaborative_filtering.itembased import CFItemBased
from recommenders.distance_based_recommender import DistanceBasedRecommender
from recommenders.content_based.content_based import ContentBasedRecommender
import data.data as d
from utils import log
from inout import importexport
pyximport.install(setup_args={"script_args":[],
                              "include_dirs":np.get_include()},
                  reload_support=True)

class CFW(RecommenderBase):
    """
    it mixes a item cf similarity matrix with a cb similarity matrix by learning a set of weight for the icm
    that push the cb similarity values towards the ones of the cf (for the values in common)
    it is meant as a way to push the content based performances up

    based on: CFW: A Collaborative Filtering System Using Posteriors Over Weights Of Evidence
    """

    RECOMMENDER_NAME = "CFW_D_Similarity_Linalg"

    def __init__(self):
        self.name = "CFW_D_Similarity_Linalg"

        # best item based similarity collaborative filter
        item = CFItemBased()
        sim_item = item.fit(d.get_urm(), 600,
                            distance=DistanceBasedRecommender.SIM_SPLUS,
                            shrink=10,
                            alpha=0.25,
                            beta=0.5,
                            l=0.25,
                            c=0.5).tocsr()
        # normalization, matrix similarity values are now among 0 and 1. little push in performances
        self.S_matrix_target = sim_item/sim_item.max()

        # best similarity content based
        content = ContentBasedRecommender()
        sim_content = content.fit(d.get_urm(),
                                  d.get_icm(),
                                  k=500,
                                  distance=DistanceBasedRecommender.SIM_SPLUS,
                                  shrink=500,
                                  alpha=0.75,
                                  beta=1,
                                  l=0.5,
                                  c=0.5).tocsr()
        # normalization, matrix similarity values are now among 0 and 1. little push in performances
        self.S_matrix_contentKNN = sim_content/sim_content.max()

    def _generateTrainData_low_ram(self):
        print(self.RECOMMENDER_NAME + ": Generating train data")
        start_time_batch = time.time()

        print(self.RECOMMENDER_NAME + ": Collaborative S density: {:.2E}, nonzero cells {}".format(
            self.S_matrix_target.nnz/self.S_matrix_target.shape[0]**2, self.S_matrix_target.nnz))

        print(self.RECOMMENDER_NAME + ": Content S density: {:.2E}, nonzero cells {}".format(
            self.S_matrix_contentKNN.nnz/self.S_matrix_contentKNN.shape[0]**2, self.S_matrix_contentKNN.nnz))

        if self.normalize_similarity:
            sum_of_squared_features = np.array(self.ICM.T.power(2).sum(axis=0)).ravel()
            sum_of_squared_features = np.sqrt(sum_of_squared_features)

        num_common_coordinates = 0
        estimated_n_samples = int(self.S_matrix_contentKNN.nnz*(1+self.add_zeros_quota)*1.2)

        self.row_list = np.zeros(estimated_n_samples, dtype=np.int32)
        self.col_list = np.zeros(estimated_n_samples, dtype=np.int32)
        self.data_list = np.zeros(estimated_n_samples, dtype=np.float64)

        num_samples = 0

        for row_index in range(self.n_items):

            start_pos_content = self.S_matrix_contentKNN.indptr[row_index]
            end_pos_content = self.S_matrix_contentKNN.indptr[row_index+1]

            content_coordinates = self.S_matrix_contentKNN.indices[start_pos_content:end_pos_content]

            start_pos_target = self.S_matrix_target.indptr[row_index]
            end_pos_target = self.S_matrix_target.indptr[row_index+1]

            target_coordinates = self.S_matrix_target.indices[start_pos_target:end_pos_target]

            # Chech whether the content coordinate is associated to a non zero target value
            # If true, the content coordinate has a collaborative non-zero value
            # if false, the content coordinate has a collaborative zero value
            is_common = np.in1d(content_coordinates, target_coordinates)

            num_common_in_current_row = is_common.sum()
            num_common_coordinates += num_common_in_current_row

            for index in range(len(is_common)):

                if num_samples == estimated_n_samples:
                    dataBlock = 1000000
                    self.row_list = np.concatenate((self.row_list, np.zeros(dataBlock, dtype=np.int32)))
                    self.col_list = np.concatenate((self.col_list, np.zeros(dataBlock, dtype=np.int32)))
                    self.data_list = np.concatenate((self.data_list, np.zeros(dataBlock, dtype=np.float64)))

                if is_common[index]:
                    # If cell exists in target matrix, add its value
                    # Otherwise it will remain zero with a certain probability

                    col_index = content_coordinates[index]

                    self.row_list[num_samples] = row_index
                    self.col_list[num_samples] = col_index

                    new_data_value = self.S_matrix_target[row_index, col_index]

                    if self.normalize_similarity:
                        new_data_value *= sum_of_squared_features[row_index]*sum_of_squared_features[col_index]

                    self.data_list[num_samples] = new_data_value

                    num_samples += 1

                elif np.random.rand() <= self.add_zeros_quota:

                    col_index = content_coordinates[index]

                    self.row_list[num_samples] = row_index
                    self.col_list[num_samples] = col_index
                    self.data_list[num_samples] = 0.0

                    num_samples += 1

            if time.time() - start_time_batch > 30 or num_samples == self.S_matrix_contentKNN.nnz*(1+self.add_zeros_quota):

                print(self.RECOMMENDER_NAME + ": Generating train data. Sample {} ( {:.2f} %) ".format(
                    num_samples, num_samples/ self.S_matrix_contentKNN.nnz*(1+self.add_zeros_quota) *100))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_batch = time.time()

        print(self.RECOMMENDER_NAME + ": Content S structure has {} out of {} ( {:.2f}%) nonzero collaborative cells".format(
            num_common_coordinates, self.S_matrix_contentKNN.nnz, num_common_coordinates/self.S_matrix_contentKNN.nnz*100))

        # Discard extra cells at the left of the array
        self.row_list = self.row_list[:num_samples]
        self.col_list = self.col_list[:num_samples]
        self.data_list = self.data_list[:num_samples]

        data_nnz = sum(np.array(self.data_list)!=0)
        data_sum = sum(self.data_list)

        collaborative_nnz = self.S_matrix_target.nnz
        collaborative_sum = sum(self.S_matrix_target.data)

        print(self.RECOMMENDER_NAME + ": Nonzero collaborative cell sum is: {:.2E}, average is: {:.2E}, "
                      "average over all collaborative data is {:.2E}".format(
                      data_sum, data_sum/data_nnz, collaborative_sum/collaborative_nnz))

    def _compute_W_sparse(self):
        if self.use_incremental:
            feature_weights = self.D_incremental
        else:
            feature_weights = self.D_best

        feature_weights = np.sqrt(np.absolute(feature_weights))
        ICM_weighted = self.ICM * csr_matrix((feature_weights,
                                              (np.arange(len(feature_weights), dtype=np.int32),
                                               np.arange(len(feature_weights), dtype=np.int32))))

        self.W_sparse = ICM_weighted * ICM_weighted.T

    def fit(self, ICM=None, URM_train=None, normalize_similarity = False, add_zeros_quota = 1, loss_tolerance = 1e-6,
            iteration_limit = 30, damp_coeff=1, use_incremental=False):
        self.URM_train = URM_train
        self.ICM = ICM
        self.n_items = self.URM_train.shape[1]
        self.n_users = self.URM_train.shape[0]
        self.n_features = self.ICM.shape[1]
        self.normalize_similarity = normalize_similarity
        self.add_zeros_quota = add_zeros_quota
        self.use_incremental = use_incremental

        self._generateTrainData_low_ram()

        common_features = self.ICM[self.row_list].multiply(self.ICM[self.col_list])

        linalg_result = linalg.lsqr(common_features, self.data_list, show = True, atol=loss_tolerance,
                                    btol=loss_tolerance, iter_lim = iteration_limit, damp=damp_coeff)

        self.D_incremental = linalg_result[0].copy()
        self.D_best = linalg_result[0].copy()
        self.epochs_best = 0
        self.loss = linalg_result[3]

        self._compute_W_sparse()

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

    def get_r_hat(self):
        r_hat = d.get_empty_urm()
        user_profile_batch = self.URM_train[d.get_target_playlists()]
        r_hat[d.get_target_playlists()] = user_profile_batch.dot(self.W_sparse)
        return r_hat

    def recommend(self, userid, N=10, urm=None, filter_already_liked=True, with_scores=False, items_to_exclude=[]):
        return self.recommend_batch([userid],
                                    N=N,
                                    urm=urm,
                                    filter_already_liked=filter_already_liked,
                                    with_scores=with_scores,
                                    items_to_exclude=items_to_exclude)

    def _print(self, normalize_similarity = False, add_zeros_quota = 1, loss_tolerance = 1e-6,
            iteration_limit = 30, damp_coeff=1, use_incremental=False):
        return '{} normalize_similarity: {}, add_zeros_quota: {}, loss_tolerance: {}, iteration_limit: {}, damp_coeff: {}, use_incremental: {}'.format(
            self.name,
            normalize_similarity,
            add_zeros_quota,
            loss_tolerance,
            iteration_limit,
            damp_coeff,
            use_incremental)

    def run(self, normalize_similarity = False, add_zeros_quota = 1, loss_tolerance = 1e-6, iteration_limit = 30, 
            damp_coeff=1, use_incremental=False, export_results=True, export_r_hat=False, export_for_validation=False):
        if export_r_hat and export_for_validation:
            urm = d.get_urm_train()
        else:
            urm = d.get_urm()
        
        self.fit(ICM=d.get_icm(),
                 URM_train=urm,
                 normalize_similarity=normalize_similarity,
                 add_zeros_quota=add_zeros_quota,
                 loss_tolerance=loss_tolerance,
                 iteration_limit=iteration_limit,
                 damp_coeff=damp_coeff,
                 use_incremental=use_incremental)
        if export_results:
            print('exporting results')
            recs = self.recommend_batch(d.get_target_playlists(),
                                         N=10,
                                         urm=urm,
                                         filter_already_liked=True,
                                         with_scores=False,
                                         items_to_exclude=[],
                                         verbose=False)
            importexport.exportcsv(recs, 'submission', self._print(normalize_similarity=normalize_similarity,
                                                                   add_zeros_quota=add_zeros_quota,
                                                                   loss_tolerance=loss_tolerance,
                                                                   iteration_limit=iteration_limit,
                                                                   damp_coeff=damp_coeff,
                                                                   use_incremental=use_incremental
                                                                   ))
        elif export_r_hat:
            print('saving estimated urm')
            self.save_r_hat(export_for_validation)

    def validate(self, user_ids=d.get_target_playlists(), log_path=None, normalize_similarity = [False], damp_coeff=[1],
                 add_zeros_quota = [1], loss_tolerance = [1e-6], iteration_limit = [30], use_incremental=[False]):

        if log_path != None:
            orig_stdout = sys.stdout
            f = open(log_path + '/' + self.name + ' ' + time.strftime('_%H-%M-%S') + ' ' +
                     time.strftime('%d-%m-%Y') + '.txt', 'w')
            sys.stdout = f

        for ns in normalize_similarity:
            for dc in damp_coeff:
                for adq in add_zeros_quota:
                    for lt in loss_tolerance:
                        for il in iteration_limit:
                            for ui in use_incremental:
                                print(self._print(normalize_similarity=ns,
                                                  add_zeros_quota=dc,
                                                  loss_tolerance=lt,
                                                  iteration_limit=il,
                                                  damp_coeff=dc,
                                                  use_incremental=ui))
                                self.fit(ICM=d.get_icm(),
                                         URM_train=d.get_urm_train(),
                                         normalize_similarity=ns,
                                         add_zeros_quota=adq,
                                         loss_tolerance=lt,
                                         iteration_limit=il,
                                         damp_coeff=dc,
                                         use_incremental=ui)

                                recs = self.recommend_batch(user_ids, urm=d.get_urm_train())
                                r.evaluate(recs, d.get_urm_test())
        if log_path != None:
            sys.stdout = orig_stdout
            f.close()

#0.039
r = CFW()
# r.validate(normalize_similarity=[True],
#            damp_coeff=[0.1, 1, 10, 100],
#            add_zeros_quota=[0.1, 1, 10, 100],
#            loss_tolerance=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
#            iteration_limit=[5, 10, 30, 50, 100, 200, 350, 500],
#            use_incremental=[False, True])
r.run(export_results=False, export_r_hat=True, export_for_validation=False)
