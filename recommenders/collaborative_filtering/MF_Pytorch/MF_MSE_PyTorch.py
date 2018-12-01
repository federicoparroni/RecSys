#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/07/2018

@author: Maurizio Ferrari Dacrema
"""


from recommenders.recommender_base import RecommenderBase
import sys
import numpy as np, pickle
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import log
import data.data as d

class MF_MSE_PyTorch(RecommenderBase):

    def __init__(self):
        self.name = 'mf_pytorch'

    def fit(self, URM_train, epochs=30, batch_size = 128, num_factors=10, learning_rate = 0.001,
            user_ids=None, URM_test=None, validation_every_n = 1, use_cuda = True):
        self.URM_train = URM_train
        self.n_factors = num_factors
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("MF_MSE_PyTorch: Using CUDA")
        else:
            self.device = torch.device('cpu')
            print("MF_MSE_PyTorch: Using CPU")

        from MatrixFactorization.PyTorch.MF_MSE_PyTorch_model import MF_MSE_PyTorch_model, DatasetIterator_URM

        n_users, n_items = self.URM_train.shape

        self.pyTorchModel = MF_MSE_PyTorch_model(n_users, n_items, self.n_factors).to(self.device)

        #Choose loss
        self.lossFunction = torch.nn.MSELoss(size_average=False)
        #self.lossFunction = torch.nn.BCELoss(size_average=False)

        self.optimizer = torch.optim.Adagrad(self.pyTorchModel.parameters(), lr = self.learning_rate)

        dataset_iterator = DatasetIterator_URM(self.URM_train)

        self.train_data_loader = DataLoader(dataset = dataset_iterator,
                                            batch_size = self.batch_size,
                                            shuffle = True,
                                            #num_workers = 2,
                                            )

        self._train_with_early_stopping(epochs, validation_every_n, user_ids=user_ids, URM_test=URM_test)

        self.W = self.W_best.copy()
        self.H = self.H_best.copy()

        sys.stdout.flush()

    def compute_score_MF(self, user_id):
        scores_array = np.dot(self.W[user_id], self.H.T)
        return scores_array

    def recommend_batch(self, userids, urm,  N=10, filter_already_liked=True, with_scores=False, items_to_exclude=[], verbose=False):
        scores_array = self.compute_score_MF(userids)
        user_profile_batch = self.URM_train[userids]

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
        scores = self.compute_score_MF(userid)
        if filter_already_liked:
            scores = self._remove_seen_on_scores(userid, scores)

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

    def _remove_seen_on_scores(self, user_id, scores):
        assert self.URM_train.getformat() == "csr", "Recommender_Base_Class: URM_train is not CSR, this will cause errors in filtering seen items"
        seen = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]
        scores[seen] = -np.inf
        return scores

    def _train_with_early_stopping(self, epochs, validation_every_n, user_ids=None, URM_test=None):
        self.best_validation_metric = None
        convergence = False
        self._initialize_incremental_model()
        self.epochs_best = 0
        currentEpoch = 0

        while currentEpoch < epochs and not convergence:

            self._run_epoch(currentEpoch)

            # Determine whether a validaton step is required
            if (currentEpoch + 1) % validation_every_n == 0:

                if URM_test == None or user_ids == None:
                    raise ValueError("Validation cannot be performed without URM_test and user_ids!")

                print("Evaluation begins")
                self._update_incremental_model()
                recs = self.recommend_batch(user_ids, None)
                self.evaluate(recs, URM_test)

            currentEpoch += 1

    def _initialize_incremental_model(self):
        self.W_incremental = self.pyTorchModel.get_W()
        self.W_best = self.W_incremental.copy()

        self.H_incremental = self.pyTorchModel.get_H()
        self.H_best = self.H_incremental.copy()

    def _update_incremental_model(self):
        self.W_incremental = self.pyTorchModel.get_W()
        self.H_incremental = self.pyTorchModel.get_H()

        self.W = self.W_incremental.copy()
        self.H = self.H_incremental.copy()

    def _update_best_model(self):
        self.W_best = self.W_incremental.copy()
        self.H_best = self.H_incremental.copy()

    def _run_epoch(self, num_epoch):
        for num_batch, (input_data, label) in enumerate(self.train_data_loader, 0):

            if num_batch % 1000 == 0:
                print("num_batch: {}".format(num_batch))

            # On windows requires int64, on ubuntu int32
            #input_data_tensor = Variable(torch.from_numpy(np.asarray(input_data, dtype=np.int64))).to(self.device)
            input_data_tensor = Variable(input_data).to(self.device)

            label_tensor = Variable(label).to(self.device)


            user_coordinates = input_data_tensor[:,0]
            item_coordinates = input_data_tensor[:,1]

            # FORWARD pass
            prediction = self.pyTorchModel(user_coordinates, item_coordinates)

            # Pass prediction and label removing last empty dimension of prediction
            loss = self.lossFunction(prediction.view(-1), label_tensor)

            # BACKWARD pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def get_r_hat(self, load_from_file=False, path=''):
        pass

    def run(self):
        pass


m = MF_MSE_PyTorch()
m.fit(d.get_urm_train(), user_ids=d.get_target_playlists(), URM_test=d.get_urm_test())
