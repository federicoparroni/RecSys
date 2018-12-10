import numpy as np
import utils.check_matrix_format as cm
import utils.log as log
import data.data as data
import pyximport
pyximport.install(setup_args={"script_args": [],
                              "include_dirs": np.get_include()},
                  reload_support=True)
from recommenders.collaborative_filtering.cython.FunkSVD_sgd import FunkSVD_sgd

class FunkSVD(object):
    '''
    FunkSVD model
    Reference: http://sifter.org/~simon/journal/20061211.html
    Factorizes the rating matrix R into the dot product of two matrices U and V of latent factors.
    U represent the user latent factors, V the item latent factors.
    The model is learned by solving the following regularized Least-squares objective function with Stochastic Gradient Descent
    \operatornamewithlimits{argmin} \limits_{U,V}\frac{1}{2}||R - UV^T||^2_2 + \frac{\lambda}{2}(||U||^2_F + ||V||^2_F)
    Latent factors are initialized from a Normal distribution with given mean and std.
    '''

    # TODO: add global effects
    def __init__(self):
        self.name = "FunkSVD"

    def fit(self, urm,
                num_factors=100,
                 learning_rate=0.2,
                 reg=0.01,
                 epochs=10,
                 init_mean=0.1,
                 init_std=0.0,
                 lrate_decay=0,
                 rnd_seed=42):
        """
        Initialize the model
        :param num_factors: number of latent factors
        :param learning_rate: initial learning rate used in SGD
        :param reg: regularization term
        :param epochs: number of iterations in training the model with SGD
        :param init_mean: mean used to initialize the latent factors
        :param init_std: standard deviation used to initialize the latent factors
        :param lrate_decay: learning rate decay
        :param rnd_seed: random seed
        """

        self.urm = urm
        self.num_factors = num_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.epochs = epochs
        self.init_mean = init_mean
        self.init_std = init_std
        self.lrate_decay = lrate_decay
        self.rnd_seed = rnd_seed

        self.U, self.V = FunkSVD_sgd(self.urm, self.num_factors, self.learning_rate, self.reg, self.epochs, self.init_mean,
                                     self.init_std,
                                     self.lrate_decay, self.rnd_seed)

    def recommend_batch(self, userids, N=10, filter_already_liked=True):
        # compute the scores using the dot product
        user_profile_batch = self.urm[userids]

        scores_array = np.dot(self.U.base[userids], self.V.base.T)

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

rec = FunkSVD()
rec.fit(urm=data.get_urm_train_1())
recs = rec.recommend_batch(userids=data.get_target_playlists())
rec.evaluate(recs, data.get_urm_test_1())
