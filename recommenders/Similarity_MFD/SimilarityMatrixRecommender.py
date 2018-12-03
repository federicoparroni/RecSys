
import pickle
import numpy as np
import utils.log as log

class SimilarityMatrixRecommender(object):
    """
    This class refers to a Recommender KNN which uses a Similarity_MFD matrix, it provides two function to compute item's score
    bot for user-based and Item-based models as well as a function to save the W_matrix
    """

    def __init__(self):
        super(SimilarityMatrixRecommender, self).__init__()

        self.sparse_weights = True

        self.compute_item_score = self.compute_score_item_based

    def compute_score_item_based(self, user_id):

        if self.sparse_weights:
            user_profile = self.URM_train[user_id]

            return user_profile.dot(self.W_sparse).toarray()

        else:

            assert False

            user_profile = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]
            user_ratings = self.URM_train.data[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]

            relevant_weights = self.W[user_profile]
            return relevant_weights.T.dot(user_ratings)

    def compute_score_user_based(self, user_id):

        if self.sparse_weights:

            return self.W_sparse[user_id].dot(self.URM_train).toarray()

        else:
            # Numpy dot does not recognize sparse matrices, so we must
            # invoke the dot function on the sparse one
            return self.URM_train.T.dot(self.W[user_id])

    def saveModel(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        dictionary_to_save = {"sparse_weights": self.sparse_weights}


        if self.sparse_weights:
            dictionary_to_save["W_sparse"] = self.W_sparse

        else:
            dictionary_to_save["W"] = self.W


        pickle.dump(dictionary_to_save,
                    open(folder_path + file_name, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)


        print("{}: Saving complete".format(self.RECOMMENDER_NAME))

    def recommend_batch(self, type, userids, filter_seen_matrix=None, N=10, filter_already_liked=True):
        """
        :param type: USER or ITEM
        """
        if filter_seen_matrix is None:
            user_profile_batch = self.URM_train[userids]
        else:
            user_profile_batch = filter_seen_matrix[userids]

        if type not in ['USER', 'ITEM']:
            log.error('type must be or USER or ITEM')
        if type == 'USER':
            scores_array = self.compute_score_user_based(userids)
        else:
            scores_array = self.compute_score_item_based(userids)

        """
        To exclude already_liked items perform a boolean indexing and replace their score with -inf
        Seen items will be at the bottom of the list but there is no guarantee they'll NOT be
        recommended
        """

        if filter_already_liked:
            scores_array[user_profile_batch.nonzero()] = -np.inf

        # magic code 🔮 to take the top N recommendations
        ranking = np.zeros((scores_array.shape[0], N), dtype=np.int)
        for row_index in range(scores_array.shape[0]):
            scores = scores_array[row_index]
            relevant_items_partition = (-scores).argpartition(N)[0:N]
            relevant_items_partition_sorting = np.argsort(-scores[relevant_items_partition])
            ranking[row_index] = relevant_items_partition[relevant_items_partition_sorting]
        """
        add target id in a way that recommendations is a list as follows
        [ [playlist1_id, id1, id2, ....., id10], ...., [playlist_id2, id1, id2, ...] ]
        """
        np_target_id = np.array(userids)
        target_id_t = np.reshape(np_target_id, (len(np_target_id), 1))
        recommendations = np.concatenate((target_id_t, ranking), axis=1)

        return recommendations

    def evaluate(self, recommendations, test_urm, at_k=10, verbose=True):
        """
        Return the MAP@k evaluation for the provided recommendations
        computed with respect to the test_urm

        Parameters
        ----------
        recommendations : list
            List of recommendations, where a recommendation
            is a list (of length N+1) of playlist_id and N items_id:
                [   [7,   18,11,76, ...] ,
                    [13,  65,83,32, ...] ,
                    [25,  30,49,65, ...] , ... ]
        test_urm : csr_matrix
            A sparse matrix
        at_k : int, optional
            The number of items to compute the precision at

        Returns
        -------
        MAP@k: (float) MAP for the provided recommendations
        """
        if not at_k > 0:
            log.error('Invalid value of k {}'.format(at_k))
            return

        aps = 0.0
        for r in recommendations:
            row = test_urm.getrow(r[0]).indices
            m = min(at_k, len(row))

            ap = 0.0
            n_elems_found = 0.0
            for j in range(1, m + 1):
                if r[j] in row:
                    n_elems_found += 1
                    ap = ap + n_elems_found / j
            if m > 0:
                ap = ap / m
                aps = aps + ap

        result = aps / len(recommendations)
        if verbose:
            log.warning('MAP: {}'.format(result))
        return result

