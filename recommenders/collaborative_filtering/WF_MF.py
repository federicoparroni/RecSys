
import numpy
import numpy as np
import sys
import data.data as data
import utils.log as log
import pyximport
from recommenders.collaborative_filtering.cython.WF_MF import factor_matrix
pyximport.install(setup_args={"script_args": [],
                              "include_dirs": np.get_include()},
                  reload_support=True)

class ProductRecommender(object):
    """
    Generates recommendations using the matrix factorization approach.
    Derived and implemented from the Netflix paper.
    Author: William Falcon
    Has 2 modes:
    Mode A: Derives P, Q matrices intrinsically for k features.
    Use this approach to learn the features.
    Mode B: Derives P matrix given a constant P matrix (Products x features). Use this if you want to
    try the approach of selecting the features yourself.
    Example 1:
    from matrix_factor_model import ProductRecommender
    modelA = ProductRecommender()
    data = [[1,2,3], [0,2,3]]
    modelA.fit(data)
    model.predict_instance(1)
    # prints array([ 0.9053102 ,  2.02257811,  2.97001565])
    Model B example
    modelB = ProductRecommender()
    data = [[1,2,3], [0,2,3]]
    # product x features
    Q = [[2,3], [2, 4], [5, 9]]
    # fit
    modelA.fit(data, Q)
    model.predict_instance(1)
    # prints array([ 0.9053102 ,  2.02257811,  2.97001565])
    """

    def __init__(self):
        self.Q = None
        self.P = None

    def fit(self, user_x_product, latent_features_guess=150, learning_rate=0.0002, steps=10, regularization_penalty=0.02, convergeance_threshold=0.001):
        """
        Trains the predictor with the given parameters.
        :param user_x_product:
        :param latent_features_guess:
        :param learning_rate:
        :param steps:
        :param regularization_penalty:
        :param convergeance_threshold:
        :return:
        """
        self.urm = user_x_product
        user_x_product = user_x_product.todense()
        print('training model...')
        self.Q, self.P = factor_matrix(user_x_product, latent_features_guess, learning_rate, steps,
                                    regularization_penalty, convergeance_threshold)

    #
    # def __factor_matrix(self, R, K, alpha, steps, beta, error_limit):
    #     """
    #     R = user x product matrix
    #     K = latent features count (how many features we think the model should derive)
    #     alpha = learning rate
    #     beta = regularization penalty (minimize over/under fitting)
    #     step = logistic regression steps
    #     error_limit = algo finishes when error reaches this level
    #     Returns:
    #     P = User x features matrix. (How strongly a user is associated with a feature)
    #     Q = Product x feature matrix. (How strongly a product is associated with a feature)
    #     To predict, use dot product of P, Q
    #     """
    #     # Transform regular array to numpy array
    #     R = R.todense()
    #
    #     # Generate P - N x K
    #     # Use random values to start. Best performance
    #     N = R.shape[0]
    #     M = R.shape[1]
    #     P = numpy.random.rand(N, K)
    #
    #     # Generate Q - M x K
    #     # Use random values to start. Best performance
    #     Q = numpy.random.rand(M, K)
    #     Q = Q.T
    #
    #     error = 0
    #
    #     # iterate through max # of steps
    #     for step in range(steps):
    #         print('performing the {} step'.format(step))
    #
    #         # iterate each cell in r
    #         for i in range(R.shape[0]):
    #             for j in range(R.shape[1]):
    #                 if R[i, j] > 0:
    #                     print(i)
    #
    #
    #                     # get the eij (error) side of the equation
    #                     eij = R[i, j] - numpy.dot(P[i, :], Q[:, j])
    #
    #                     for k in range(K):
    #                         # (*update_rule) update pik_hat
    #                         P[i, k] = P[i, k] + alpha * (2 * eij * Q[k, j] - beta * P[i, k])
    #
    #                         # (*update_rule) update qkj_hat
    #                         Q[k, j] = Q[k, j] + alpha * (2 * eij * P[i, k] - beta * Q[k, j])
    #
    #         # Measure error
    #         print('calculate the error')
    #         error = self.__error(R, P, Q, K, beta)
    #         print('error at iteration {} is: {}'.format(step, error))
    #
    #         # Terminate when we converge
    #         if error < error_limit:
    #             break
    #
    #     # track Q, P (learned params)
    #     # Q = Products x feature strength
    #     # P = Users x feature strength
    #     self.Q = Q.T
    #     self.P = P
    #
    #     self.__print_fit_stats(error, N, M)
    #
    # def __error(self, R, P, Q, K, beta):
    #     """
    #     Calculates the error for the function
    #     :param R:
    #     :param P:
    #     :param Q:
    #     :param K:
    #     :param beta:
    #     :return:
    #     """
    #     e = 0
    #     for i in range(R.shape[0]):
    #         for j in range(R.shape[1]):
    #             if R[i, j] > 0:
    #
    #                 # loss function error sum( (y-y_hat)^2 )
    #                 e = e + pow(R[i, j]-numpy.dot(P[i, :], Q[:, j]), 2)
    #
    #                 # add regularization
    #                 for k in range(K):
    #
    #                     # error + ||P||^2 + ||Q||^2
    #                     e = e + (beta/2) * (pow(P[i, k], 2) + pow(Q[k, j], 2))
    #     return e

    def __print_fit_stats(self, error, samples_count, products_count):
        print('training complete...')
        print('------------------------------')
        print('Stats:')
        print('Error: %0.2f' % error)
        print('Samples: ' + str(samples_count))
        print('Products: ' + str(products_count))
        print('------------------------------')

    def recommend_batch(self, userids, N=10, filter_already_liked=True):
        # compute the scores using the dot product
        user_profile_batch = self.urm[userids]

        scores_array = numpy.dot(self.P[userids], self.Q.T)

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


if __name__ == '__main__':
    rec = ProductRecommender()
    rec.fit(user_x_product=data.get_urm_train_1(), latent_features_guess=10, learning_rate=0.01, steps=2,
            regularization_penalty=0.2, convergeance_threshold=0.01)
    recs = rec.recommend_batch(data.get_target_playlists())
    rec.evaluate(recommendations=recs, test_urm=data.get_urm_test_1())

