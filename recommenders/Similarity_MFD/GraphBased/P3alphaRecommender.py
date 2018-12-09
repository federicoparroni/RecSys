
import numpy as np
import scipy.sparse as sps

from sklearn.preprocessing import normalize
import time, sys
import data.data as data
import utils.log as log




class P3alphaRecommender():
    """ P3alpha recommender """

    RECOMMENDER_NAME = "P3alphaRecommender"

    def __init__(self, URM_train):
        self.URM_train = URM_train
        self.sparse_weights = True

    def fit(self, topK=500, alpha=1.7, min_rating=1, implicit=True, normalize_similarity=True):

        self.topK = topK
        self.alpha = alpha
        self.min_rating = min_rating
        self.implicit = implicit
        self.normalize_similarity = normalize_similarity


        #
        # if X.dtype != np.float32:
        #     print("P3ALPHA fit: For memory usage reasons, we suggest to use np.float32 as dtype for the dataset")

        if self.min_rating > 0:
            self.URM_train.data[self.URM_train.data < self.min_rating] = 0
            self.URM_train.eliminate_zeros()
            if self.implicit:
                self.URM_train.data = np.ones(self.URM_train.data.size, dtype=np.float32)

        #Pui is the row-normalized urm
        Pui = normalize(self.URM_train, norm='l1', axis=1)

        #Piu is the column-normalized, "boolean" urm transposed
        X_bool = self.URM_train.transpose(copy=True)
        X_bool.data = np.ones(X_bool.data.size, np.float32)
        #ATTENTION: axis is still 1 because i transposed before the normalization
        Piu = normalize(X_bool, norm='l1', axis=1)
        del(X_bool)

        # Alfa power
        if self.alpha != 1.:
            Pui = Pui.power(self.alpha)
            Piu = Piu.power(self.alpha)

        # Final matrix is computed as Pui * Piu * Pui
        # Multiplication unpacked for memory usage reasons
        block_dim = 200
        d_t = Piu

        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0


        start_time = time.time()
        start_time_printBatch = start_time

        for current_block_start_row in range(0, Pui.shape[1], block_dim):

            if current_block_start_row + block_dim > Pui.shape[1]:
                block_dim = Pui.shape[1] - current_block_start_row

            similarity_block = d_t[current_block_start_row:current_block_start_row + block_dim, :] * Pui
            similarity_block = similarity_block.toarray()

            for row_in_block in range(block_dim):
                row_data = similarity_block[row_in_block, :]
                row_data[current_block_start_row + row_in_block] = 0

                best = row_data.argsort()[::-1][:self.topK]

                notZerosMask = row_data[best] != 0.0

                values_to_add = row_data[best][notZerosMask]
                cols_to_add = best[notZerosMask]

                for index in range(len(values_to_add)):

                    if numCells == len(rows):
                        rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                        cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                        values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))


                    rows[numCells] = current_block_start_row + row_in_block
                    cols[numCells] = cols_to_add[index]
                    values[numCells] = values_to_add[index]

                    numCells += 1


            if time.time() - start_time_printBatch > 60:
                print("Processed {} ( {:.2f}% ) in {:.2f} minutes. Rows per second: {:.0f}".format(
                    current_block_start_row,
                    100.0 * float(current_block_start_row) / Pui.shape[1],
                    (time.time() - start_time) / 60,
                    float(current_block_start_row) / (time.time() - start_time)))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()

        self.W_sparse = sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])), shape=(Pui.shape[1], Pui.shape[1]))

        if self.normalize_similarity:
            self.W_sparse = normalize(self.W_sparse, norm='l1', axis=1)

        if self.topK != False:
            self.W_sparse = similarityMatrixTopK(self.W_sparse, k=self.topK)
            self.sparse_weights = True

    def recommend_batch(self, userids, N=10, filter_already_liked=True, verbose=False):
        """
        look for comment on superclass method
        """
        # compute the scores using the dot product
        user_profile_batch = self.URM_train[userids]

        scores_array = np.dot(user_profile_batch, self.W_sparse)

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
            scores = scores.todense()
            relevant_items_partition = (-scores).argpartition(N)[0, 0:N]
            relevant_items_partition_sorting = np.argsort(-scores[0, relevant_items_partition])
            ranking[row_index] = relevant_items_partition[0, relevant_items_partition_sorting[0, 0:N]]

        """
        add target id in a way that recommendations is a list as follows
        [ [playlist1_id, id1, id2, ....., id10], ...., [playlist_id2, id1, id2, ...] ]
        """
        np_target_id = np.array(userids)
        target_id_t = np.reshape(np_target_id, (len(np_target_id), 1))
        recommendations = np.concatenate((target_id_t, ranking), axis=1)

        return recommendations


def similarityMatrixTopK(item_weights, k=100, verbose=False, inplace=True):
    """
    The function selects the TopK most similar elements, column-wise
    :param item_weights:
    :param forceSparseOutput:
    :param k:
    :param verbose:
    :param inplace: Default True, WARNING matrix will be modified
    :return:
    """

    assert (item_weights.shape[0] == item_weights.shape[1]), "selectTopK: ItemWeights is not a square matrix"

    start_time = time.time()

    if verbose:
        print("Generating topK matrix")

    nitems = item_weights.shape[1]
    k = min(k, nitems)

    # for each column, keep only the top-k scored items
    sparse_weights = not isinstance(item_weights, np.ndarray)

    W=item_weights

    W_sparse = sps.csr_matrix(W, shape=(nitems, nitems))

    return W_sparse

def evaluate(recommendations, test_urm, at_k=10, verbose=True):
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

rec = P3alphaRecommender(data.get_urm_train())
rec.fit()

r_hat = sps.csr_matrix(np.dot(rec.URM_train[data.get_target_playlists()], rec.W_sparse))
sps.save_npz('raw_data/saved_r_hat_evaluation/P3alpha', r_hat)

#recs = rec.recommend_batch(data.get_target_playlists())
#evaluate(recs, test_urm=data.get_urm_test())