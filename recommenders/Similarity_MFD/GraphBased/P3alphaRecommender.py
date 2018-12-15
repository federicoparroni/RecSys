
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import normalize
import time, sys
import data.data as data
import utils.log as log
import inout.importexport as export
from bayes_opt import BayesianOptimization


class P3alphaRecommender():
    """ P3alpha recommender """

    RECOMMENDER_NAME = "P3alpha"

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
        return np.concatenate((target_id_t, ranking), axis=1)


    def validateStep(self, k, min_rating, alpha):
        # gather saved parameters from self
        targetids = self._validation_dict['targetids']
        urm_test = self._validation_dict['urm_test']
        N = self._validation_dict['N']
        normalize_similarity = self._validation_dict['normalize_similarity']
        implicit = self._validation_dict['implicit']
        verbose = self._validation_dict['verbose']

        self.fit(topK=int(k), alpha=alpha, min_rating=int(min_rating), implicit=implicit, normalize_similarity=normalize_similarity)
        # evaluate the model with the current weigths
        recs = self.recommend_batch(userids=targetids, N=N, verbose=verbose)
        return evaluate(recs, test_urm=urm_test)

    def validate(self, iterations, urm_test, targetids, normalize_similarity=True,
                k=(50,900), min_rating=(0,2), alpha=(0,2), N=10, implicit=True, verbose=False):
        
        # save the params in self to collect them later
        self._validation_dict = {
            'urm_test': urm_test,
            'targetids': targetids,
            'normalize_similarity': normalize_similarity,
            'N': N,
            'implicit': implicit,
            'verbose': verbose
        }

        pbounds = {
            'k': k if isinstance(k, tuple) else (int(k),int(k)),
            'min_rating': min_rating if isinstance(min_rating, tuple) else (int(min_rating),int(min_rating)),
            'alpha': alpha if isinstance(alpha, tuple) else (float(alpha),float(alpha))
        }

        optimizer = BayesianOptimization(
            f=self.validateStep,
            pbounds=pbounds,
            random_state=1
        )
        optimizer.maximize(
            init_points=2,
            n_iter=iterations
        )

        log.warning('Max found: {}'.format(optimizer.max))
        return optimizer


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

"""
If this file is executed, test the P3alpha recommender
"""
if __name__ == '__main__':
    print()
    log.success('++ What do you want to do? ++')
    log.warning('(t) Test the model with some default params')
    log.warning('(r) Save the R^')
    log.warning('(s) Save the similarity matrix')
    log.warning('(v) Validate the model')
    log.warning('(e) Export the submission')
    log.warning('(x) Exit')
    arg = input()[0]
    print()
    
    if arg == 't':
        model = P3alphaRecommender(data.get_urm_train_1())
        model.fit(topK=900, alpha=1.2, min_rating=0, implicit=True, normalize_similarity=False)
        recs = model.recommend_batch(data.get_target_playlists())
        evaluate(recs, test_urm=data.get_urm_test_1())
    elif arg == 'r':
        log.info('Wanna save for evaluation (y/n)?')
        if input()[0] == 'y':
            model = P3alphaRecommender(data.get_urm())
            path = 'raw_data/saved_r_hat_evaluation/'
        else:
            model = P3alphaRecommender(data.get_urm_train_1())
            path = 'raw_data/saved_r_hat/'

        model.fit(topK=500, alpha=1.7, min_rating=1, normalize_similarity=True)

        print('Saving the R^...')
        r_hat = sps.csr_matrix(np.dot(model.URM_train[data.get_target_playlists()], model.W_sparse))
        sps.save_npz(path + model.RECOMMENDER_NAME, r_hat)
    elif arg == 's':
        model.fit(topK=500, alpha=1.7, min_rating=1, normalize_similarity=True)
        print('Saving the similarity matrix...')
        sps.save_npz('raw_data/saved_sim_matrix_evaluation_1/{}'.format(model.RECOMMENDER_NAME), model.W_sparse)
    elif arg == 'v':
        model = P3alphaRecommender(data.get_urm_train_1())
        model.validate(iterations=15, urm_test=data.get_urm_test_1(), targetids=data.get_target_playlists(),
                    normalize_similarity=True, k=(50,900), min_rating=(0,2), alpha=(0,2), verbose=False)
    elif arg == 'e':
        model = P3alphaRecommender(data.get_urm())
        model.fit(topK=900, alpha=1.2, min_rating=0, implicit=True, normalize_similarity=False)
        recs = model.recommend_batch(data.get_target_playlists())
        export.exportcsv(recs, name=model.RECOMMENDER_NAME)
    elif arg == 'x':
        pass
    else:
        log.error('Wrong option!')
