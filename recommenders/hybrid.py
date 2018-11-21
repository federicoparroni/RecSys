from recommenders.recommender_base import RecommenderBase
import numpy as np


class Hybrid(RecommenderBase):
    """
    recommender builded passing to the init method an array of extimated R (r_hat_array) that will be combined to obtain an hybrid r_hat
    """

    def __init__(self, r_hat_array, urm):
        self.r_hat_array = r_hat_array
        self.normalized = False
        self.hybrid_r_hat = None
        self.urm = urm

    def normalize_max_row(self, userids):
        """
        :param userids: user for which compute the predictions
        :return: return an array containing the normalized matrices
        """
        normalized_r_hat_array = []
        for r in self.r_hat_array:
            filterd_r = r[userids]
            for row_index in range(filterd_r.shape[0]):
                row = filterd_r[row_index]
                max_row = row.max()
                normalized_row = row/max_row
                filterd_r[row_index] = normalized_row
            normalized_r_hat_array.append(filterd_r)
        return normalized_r_hat_array

    def normalize_max_matrix(self, userids):
        """
        :param userids: user for which compute the predictions
        :return: return an array containing the normalized matrices
        """
        normalized_r_hat_array = []
        for r in self.r_hat_array:
            max_matrix = r.max()
            r_filtered = r[userids]
            normalized_matrix = r_filtered/max_matrix
            normalized_r_hat_array.append(normalized_matrix)
        return normalized_r_hat_array

    def recommend_batch(self, userids, normalization_mode=0, N=10, filter_already_liked=True, items_to_exclude=[], verbose=False):
        """
        method used for get the hybrid prediction from the r_hat_matrices passed as parameter during the creation of the recommender
        step1: normalize the matrices, there are two type of normalization
        step2: sum the normalized matrices
        step3: get the prediction from the matrix obtained as sum

        :param userids: user for which compute the predictions
        :param normalization_mode: 0(default) normalize each row for the max of the row
                                    1 normalize each row for the max of the matrix
        :param N: how many items to predict for each user
        :param filter_already_liked: filter the element with which the user has interacted
        :param items_to_exclude: list of item to exclude from the predictions
        :param verbose:
        :return:
        """
        # STEP1
        if normalization_mode==0:
            normalized_r_hat_array = self.normalize_max_row(userids)
        else:
            normalized_r_hat_array = self.normalize_max_matrix(userids)
        hybrid_r_hat = np.zeros((normalized_r_hat_array[0].shape))

        # STEP2
        for m in normalized_r_hat_array:
            hybrid_r_hat += m

        #filter seen elements
        if filter_already_liked:
            user_profile = self.urm[userids]
            hybrid_r_hat[user_profile.nonzero()] = -np.inf

        # STEP3

        ranking = np.zeros((hybrid_r_hat.shape[0], N), dtype=np.int)
        hybrid_r_hat = hybrid_r_hat.todense()

        for row_index in range(hybrid_r_hat.shape[0]):
            scores_row = hybrid_r_hat[row_index]

            relevant_items_partition = (-scores_row).argpartition(N)[0, 0:N]
            relevant_items_partition_sorting = np.argsort(-scores_row[0, relevant_items_partition])
            ranking[row_index] = relevant_items_partition[0, relevant_items_partition_sorting[0, 0:N]]

        recommendations = self._insert_userids_as_first_col(userids, ranking)
        return recommendations







