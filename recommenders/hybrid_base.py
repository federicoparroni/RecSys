from recommenders.recommender_base import RecommenderBase
import numpy as np
import data.data as data
import scipy.sparse as sps
import utils.log as log
import time
from bayes_opt import BayesianOptimization
import sklearn.preprocessing as sk
import utils.check_matrix_format as cm



class Hybrid(RecommenderBase):
    """
    recommender builded passing to the init method an array of extimated R (matrices_array) that will be combined to obtain an hybrid r_hat
    """
    MAX_MATRIX = 'MAX_MATRIX'
    MAX_ROW = 'MAX_ROW'
    L2 = 'L2'
    NONE = 'NONE'

    def __init__(self, urm_filter_tracks):

        self.urm_filter_tracks = urm_filter_tracks

        index = 0
        for r in self.matrices_array:
            self.matrices_array[index] = cm.check_matrix(r, 'csr')
            index += 1

        self._normalization(normalization_mode=self.normalization_mode)

        print('matrices_normalized')

    def normalize_max_row(self):
        """
        :param userids: user for which compute the predictions
        :return: return an array containing the normalized matrices
        """
        normalized_matrices_array = []
        count = 0
        for r in self.matrices_array:
            for row_index in range(r.shape[0]):

                print(row_index*100/r.shape[0])

                row = r.getrow(row_index)
                max_row = row.max()
                r[row_index].data = r[row_index].data/max_row


                #row = r[row_index]
                #max_row = row.max()
                #normalized_row = row/max_row
                #r[row_index] = normalized_row
            normalized_matrices_array.append(r)
            count+=1
        return normalized_matrices_array

    def normalize_max_matrix(self):
        """
        :param userids: user for which compute the predictions
        :return: return an array containing the normalized matrices
        """
        normalized_matrices_array = []
        count = 0
        for r in self.matrices_array:

            #let the values positives
            #min = r.min()
            #print(min)
            #if min < 0:
            #    r.data = r.data - min

            #avg = np.sum(r.data) / len(r.data)
            #print('avg: {}'.format(avg))
            #r.data = r.data - avg

            max_matrix = r.max()
            print('max: {}'.format(max_matrix))

            #print('avg/max: {}'.format(avg/max_matrix))

            #r.data = r.data*(avg/max_matrix)

            r.data = r.data / max_matrix

            #adding confidence
            #r.data=r.data*(r.data-avg)*100
            #for ind in range(len(r.data)):
            #    confidence = (r.data[ind]-avg)*100
            #    r.data[ind] = r.data[ind]*confidence

            normalized_matrices_array.append(r)
            count += 1
        return normalized_matrices_array

    def normalize_l2(self):
        normalized_matrices_array = []
        for r in self.matrices_array:
            r = sk.normalize(r)
            normalized_matrices_array.append(r)
        return normalized_matrices_array

    def _normalization(self, normalization_mode):
        if normalization_mode=='MAX_ROW':
            self.normalized_matrices_array = self.normalize_max_row()
        elif normalization_mode=='MAX_MATRIX':
            self.normalized_matrices_array = self.normalize_max_matrix()
        elif normalization_mode == 'NONE':
            self.normalized_matrices_array = self.matrices_array
        elif normalization_mode == 'L2':
            self.normalized_matrices_array = self.normalize_l2()
        else:
            log.error('invalid string for normalization')
            return

    def recommend_batch(self, weights_array, target_userids, N=10, filter_already_liked=True,
                        items_to_exclude=[], verbose=False):
        """
        method used for get the hybrid prediction from the matrices passed as parameter during the creation of the recommender
        step1: normalize the matrices, there are two type of normalization
        step2: sum the normalized matrices
        step3: get the prediction from the matrix obtained as sum

        :param weights_array: array of the weights of each r_hat matrix
        :param userids: user for which compute the predictions
        :param normalization_mode: 'MAX_ROW'(default) normalize each row for the max of the row
                                    'MAX_MATRIX' normalize each row for the max of the matrix
        :param N: how many items to predict for each user
        :param filter_already_liked: filter the element with which the user has interacted
        :param items_to_exclude: list of item to exclude from the predictions
        :param verbose:
        :return:
        """

        start = time.time()

        hybrid_matrix = sps.csr_matrix(self.normalized_matrices_array[0].shape)

        count = 0
        for m in self.normalized_matrices_array:
            hybrid_matrix += m*weights_array[count]
            count += 1


        if self.name == 'HybridSimilarity':
            #compute the r_hat if we have the similarity
            if self.INVERSE == False:
                hybrid_matrix = self.urm_filter_tracks.dot(hybrid_matrix)
            else:
                hybrid_matrix = hybrid_matrix.dot(self.urm_filter_tracks)

        print('hybrid matrix created in {:.2f} s'.format(time.time() - start))

        start = time.time()
        #filter seen elements
        if filter_already_liked:
            user_profile = self.urm_filter_tracks
            #hybrid_matrix = hybrid_matrix.todense()
            hybrid_matrix[user_profile.nonzero()] = -np.inf


        print('seen elements filtered in {:.2f} s'.format(time.time()-start))

        start = time.time()
        # STEP3
        ranking = np.zeros((len(target_userids), N), dtype=np.int)
        #hybrid_matrix = hybrid_matrix.todense()

        count = 0
        for row_index in target_userids:
            scores_row = hybrid_matrix.getrow(row_index)
            scores_row = scores_row.todense()

            relevant_items_partition = (-scores_row).argpartition(N)[0, 0:N]
            relevant_items_partition_sorting = np.argsort(-scores_row[0, relevant_items_partition])
            ranking[count] = relevant_items_partition[0, relevant_items_partition_sorting[0, 0:N]]
            count += 1

        print('recommendations created in {:.2f} s'.format(time.time()-start))

        return self._insert_userids_as_first_col(target_userids, ranking)

    def fit(self):
        pass

    def get_r_hat(self, weights_array):
        hybrid_matrix = data.get_empty_urm()
        count = 0
        for m in self.normalized_matrices_array:
            hybrid_matrix += m*weights_array[count]
            count += 1
        return hybrid_matrix

    def recommend(self, userid, N=10, urm=None, filter_already_liked=True, with_scores=False, items_to_exclude=[]):
        pass

    def validateStep(self, **dict):
        # gather saved parameters from self
        targetids = self._validation_dict['targetids']
        urm_test = self._validation_dict['urm_test']
        N = self._validation_dict['N']
        filter_already_liked = self._validation_dict['filter_already_liked']
        items_to_exclude = self._validation_dict['items_to_exclude']

        # build weights array from dictionary
        weights = []
        for i in range(len(dict)):
            w = dict['w{}'.format(i)]
            weights.append(w)
        
        # evaluate the model with the current weigths
        recs = self.recommend_batch(weights, target_userids=targetids, N=N,
            filter_already_liked=filter_already_liked, items_to_exclude=items_to_exclude, verbose=False)
        return self.evaluate(recs, test_urm=urm_test)

    def validate(self, iterations, urm_test, userids=None,
                        N=10, filter_already_liked=True, items_to_exclude=[], verbose=False):
        # save the params in self to collect them later
        self._validation_dict = {
            'targetids': userids,
            'urm_test': urm_test,
            'N': N,
            'filter_already_liked': filter_already_liked,
            'items_to_exclude': items_to_exclude
        }

        pbounds = {}
        for i in range(len(self.matrices_array)):
            pbounds['w{}'.format(i)] = (0,1)

        optimizer = BayesianOptimization(
            f=self.validateStep,
            pbounds=pbounds,
            random_state=1,
        )
        optimizer.maximize(
            init_points=2,
            n_iter=iterations,
        )

        print(optimizer.max)
        return optimizer

    def run(self):
        pass



