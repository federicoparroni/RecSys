from recommenders.recommender_base import RecommenderBase
import numpy as np
import data.data as data
import scipy.sparse as sps
import utils.log as log


class Hybrid(RecommenderBase):
    """
    recommender builded passing to the init method an array of extimated R (r_hat_array) that will be combined to obtain an hybrid r_hat
    """

    def __init__(self, r_hat_array, urm=data.get_urm()):
        self.r_hat_array = r_hat_array
        self.hybrid_r_hat = None
        self.urm = urm

    def normalize_max_row(self, weights_array):
        """
        :param userids: user for which compute the predictions
        :return: return an array containing the normalized matrices
        """
        normalized_r_hat_array = []
        count = 0
        for r in self.r_hat_array:
            for row_index in range(r.shape[0]):

                log.progressbar(row_index, r.shape[0])
                print(row_index*100/r.shape[0])

                row = r[row_index]
                max_row = row.max()
                normalized_row = row*weights_array[count]/max_row
                r[row_index] = normalized_row
            normalized_r_hat_array.append(r)
            count+=1
        return normalized_r_hat_array

    def normalize_max_matrix(self, weights_array):
        """
        :param userids: user for which compute the predictions
        :return: return an array containing the normalized matrices
        """
        normalized_r_hat_array = []
        count = 0
        for r in self.r_hat_array:
            max_matrix = r.max()
            normalized_matrix = (r*weights_array[count]/max_matrix)
            #normalized_matrix.data *= 10
            #normalized_matrix.data **= 3
            normalized_r_hat_array.append(normalized_matrix)
            count += 1
        return normalized_r_hat_array

    def recommend_batch(self, weights_array, userids=data.get_target_playlists(), normalization_mode='MAX_MATRIX',
                        N=10, filter_already_liked=True, items_to_exclude=[], verbose=False):
        """
        method used for get the hybrid prediction from the r_hat_matrices passed as parameter during the creation of the recommender
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
        # STEP1
        if normalization_mode=='MAX_ROW':
            normalized_r_hat_array = self.normalize_max_row(weights_array)
        elif normalization_mode=='MAX_MATRIX':
            normalized_r_hat_array = self.normalize_max_matrix(weights_array)
        else:
            log.error('invalid string for normalization')
            return

        print('matrices normalized')

        hybrid_r_hat = sps.csr_matrix(np.zeros((normalized_r_hat_array[0].shape)))

        # STEP2
        for m in normalized_r_hat_array:
            hybrid_r_hat += m

        #filter seen elements
        if filter_already_liked:
            user_profile = self.urm[userids]
            hybrid_r_hat[user_profile.nonzero()] = -np.inf

        """
        # r_hat has only the row for the target playlists, so recreate a matrix with a shape = to the shape of the original_urm
        reconstructed_r_hat = sps.csr_matrix(self.urm.shape)
        reconstructed_r_hat[data.get_target_playlists()] = hybrid_r_hat
        """

        # STEP3
        ranking = np.zeros((len(userids), N), dtype=np.int)
        hybrid_r_hat = hybrid_r_hat.todense()

        for row_index in range(len(userids)):
            scores_row = hybrid_r_hat[row_index]

            relevant_items_partition = (-scores_row).argpartition(N)[0, 0:N]
            relevant_items_partition_sorting = np.argsort(-scores_row[0, relevant_items_partition])
            ranking[row_index] = relevant_items_partition[0, relevant_items_partition_sorting[0, 0:N]]

        print('recommendations created')

        return self._insert_userids_as_first_col(userids, ranking)


    def fit(self):
        pass

    def get_r_hat(self, load_from_file=False, path=''):
        pass

    def recommend(self, userid, N=10, urm=None, filter_already_liked=True, with_scores=False, items_to_exclude=[]):
        pass

    def run(self):
        pass







