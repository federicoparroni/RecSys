import numpy as np
import utils.check_matrix_format as cm
import utils.log as log
import data.data as data
import pyximport
pyximport.install(setup_args={"script_args": [],
                              "include_dirs": np.get_include()},
                  reload_support=True)
from recommenders.collaborative_filtering.cython.BPRMF_sgd import BPRMF_sgd, user_uniform_item_uniform_sampling, user_uniform_item_pop_sampling

class BPRMF():
    '''
    BPRMF model
    '''

    # TODO: add global effects
    def __init__(self):
        self.name = 'BPRMF'

    def fit(self, R):
        '''
        Initialize the model
        :param num_factors: number of latent factors
        :param lrate: initial learning rate used in SGD
        :param user_reg: regularization for the user factors
        :param pos_reg: regularization for the factors of the positive sampled items
        :param neg_reg: regularization for the factors of the negative sampled items
        :param iters: number of iterations in training the model with SGD
        :param sampling_type: type of sampling. Supported types are 'user_uniform_item_uniform' and 'user_uniform_item_pop'
        :param sample_with_replacement: `True` to sample positive items with replacement (doesn't work with 'user_uniform_item_pop')
        :param use_resampling: `True` to resample at each iteration during training
        :param sampling_pop_alpha: float smoothing factor for popularity based samplers (e.g., 'user_uniform_item_pop')
        :param init_mean: mean used to initialize the latent factors
        :param init_std: standard deviation used to initialize the latent factors
        :param lrate_decay: learning rate decay
        :param rnd_seed: random seed
        :param verbose: controls verbosity in output
        '''
        self.dataset = R
        R = cm.check_matrix(R, 'csr', dtype=np.float32)
        self.X, self.Y = BPRMF_sgd(R,
                                   num_factors=100,
                                   lrate=0.1,
                                   user_reg=0.0015,
                                   pos_reg=0.0015,
                                   neg_reg=0.0015,
                                   iters=10,
                                   sampling_type='user_uniform_item_uniform',
                                   sample_with_replacement=True,
                                   use_resampling=True,
                                   sampling_pop_alpha=1.0,
                                   init_mean=0.0,
                                   init_std=0.1,
                                   lrate_decay=1.0,
                                   rnd_seed=42,
                                   verbose=True)

    def recommend_batch(self, userids, N=10, filter_already_liked=True):
        # compute the scores using the dot product
        user_profile_batch = self.dataset[userids]

        scores_array = np.dot(self.X[userids], self.Y.T)

        """
        To exclude already_liked items perform a boolean indexing and replace their score with -inf
        Seen items will be at the bottom of the list but there is no guarantee they'll NOT be
        recommended
        """
        # compute the scores using the dot product

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
            for j in range(1, m+1):
                if r[j] in row:
                    n_elems_found += 1
                    ap = ap + n_elems_found/j
            if m > 0:
                ap = ap/m
                aps = aps + ap

        result = aps/len(recommendations)
        if verbose:
            log.warning('MAP: {}'.format(result))
        return result

rec = BPRMF()
rec.fit(R=data.get_urm_train_1(), )
recs = rec.recommend_batch(data.get_target_playlists())
rec.evaluate(recs, data.get_urm_test_1())

