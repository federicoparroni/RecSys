import numpy as np
import implicit # The Cython library
from recommenders.recommender_base import RecommenderBase
import scipy.sparse as sps
from utils import log
import data.data as data
from inout.importexport import exportcsv
import utils.log as log
import time
import os


class AlternatingLeastSquare(RecommenderBase):

    """
    Reference: http://yifanhu.net/PUB/collaborative_filtering.pdf (PAPER)
               https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe (SIMPLE EXPLANATION)

    Implementation of Alternating Least Squares with implicit data. We iteratively
    compute the user (x_u) and item (y_i) vectors using the following formulas:

    x_u = ((Y.T*Y + Y.T*(Cu - I) * Y) + lambda*I)^-1 * (X.T * Cu * p(u))
    y_i = ((X.T*X + X.T*(Ci - I) * X) + lambda*I)^-1 * (Y.T * Ci * p(i))

    [link text](http://www.example.com)
    """
    def __init__(self, URM):
        self.urm = URM
        self.name = 'ALS'

    def get_r_hat(self, load_from_file=False, path=''):
        """
        compute the r_hat for the model
        :return  r_hat
        """
        if load_from_file:
            r_hat = sps.load_npz(path)
        else:
            if self.user_vecs is None:
                log.error('the recommender has not been trained, call the fit() method')
            s_user_vecs = sps.csr_matrix(self.user_vecs)
            s_item_vecs_t = sps.csr_matrix(self.item_vecs.T)
            r_hat = s_user_vecs[data.get_target_playlists()].dot(s_item_vecs_t)
        return r_hat

    def fit(self, urm_train=data.get_urm(), factors=550, regularization=0.15, iterations=300, alpha=25):
        """
        train the model finding the two matrices U and V: U*V.T=R  (R is the extimated URM)

        Parameters
        ----------
        :param (csr) urm_train: The URM matrix of shape (number_users, number_items).
        :param (int) factors: How many latent features we want to compute.
        :param (float) regularization: lambda_val regularization value
        :param (int) iterations: How many times we alternate between fixing and updating our user and item vectors
        :param (int) alpha: The rate in which we'll increase our confidence in a preference with more interactions.

        Returns
        -------
        :return (csr_matrix) user_vecs: matrix N_user x factors
        :return (csr_matrix) item_vecs: matrix N_item x factors

        """
        self.urm = urm_train
        sparse_item_user = self.urm.T

        # Initialize the als model and fit it using the sparse item-user matrix
        self._model = implicit.als.AlternatingLeastSquares(factors=factors,
                                                          regularization=regularization,
                                                          iterations=iterations)

        # Calculate the confidence by multiplying it by our alpha value.
        data_conf = (sparse_item_user*alpha).astype('double')

        # Fit the model
        self._model.fit(data_conf)

        # set the user and item vectors for our model R = user_vecs * item_vecs.T
        self.user_vecs = self._model.user_factors
        self.item_vecs = self._model.item_factors


    def recommend(self, userid, N=10, urm=None, filter_already_liked=True, with_scores=True, items_to_exclude=[]):
        """
        look for comment on superclass method
        """
        # compute the scores using the dot product
        user_profile = self.urm[userid]

        scores = np.dot(self.user_vecs[userid], self.item_vecs.T)

        # To exclude seen items perform a boolean indexing and replace their score with -inf
        # Seen items will be at the bottom of the list but there is no guarantee they'll NOT be
        # recommended
        if filter_already_liked:
            scores[user_profile.nonzero()[1]] = -np.inf

        relevant_items_partition = np.argpartition((-scores), N)[0:N]
        relevant_items_partition_sorting = np.argsort(-scores[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]

        return ranking

    def recommend_batch(self, userids, N=10, filter_already_liked=True, with_scores=False, items_to_exclude=[], verbose=False):
        """
        look for comment on superclass method
        """
        # compute the scores using the dot product
        user_profile_batch = self.urm[userids]

        scores_array = np.dot(self.user_vecs[userids], self.item_vecs.T)

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
        
        # include userids as first column
        recommendations = self._insert_userids_as_first_col(userids, ranking)

        return recommendations


    def run(self, urm_train=None, urm=None, urm_test=None, targetids=None,
            factors=100, regularization=0.01, iterations=100, alpha=25, with_scores=False, export=True, verbose=True):
        """
        Run the model and export the results to a file

        Parameters
        ----------
        num_factors : int, number of latent factors
        urm : csr matrix, URM. If None, used: data.get_urm_train(). This should be the
            entire URM for which the targetids corresponds to the row indexes.
        urm_test : csr matrix, urm where to test the model. If None, use: data.get_urm_test()
        targetids : list, target user ids. If None, use: data.get_target_playlists()

        Returns
        -------
        recs: (list) recommendations
        map10: (float) MAP10 for the provided recommendations
        """
        _urm = data.get_urm_train()
        _icm = data.get_icm()
        _urm_test = data.get_urm_test()
        _targetids = data.get_target_playlists()
        #_targetids = data.get_all_playlists()

        start = time.time()

        urm_train = _urm if urm_train is None else urm_train
        #urm = _urm if urm is None else urm
        urm_test = _urm_test if urm_test is None else urm_test
        targetids = _targetids if targetids is None else targetids

        self.fit(urm_train=urm_train, factors=factors, regularization=regularization, iterations=iterations, alpha=alpha)
        recs = self.recommend_batch(userids=targetids, with_scores=with_scores, verbose=verbose)

        map10 = None
        if len(recs) > 0:
            map10 = self.evaluate(recs, test_urm=urm_test, verbose=verbose)
        else:
            log.warning('No recommendations available, skip evaluation')

        if export:
            exportcsv(recs, path='submission', name=self.name, verbose=verbose)

        if verbose:
            log.info('Run in: {:.2f}s'.format(time.time()-start))
        
        return recs, map10

    def test(self, num_factors=250):
        """
        Test the model without saving the results. Default distance: SPLUS
        """
        return self.run(factors=200, iterations=10)


    def validate_als(self, factors_array, regularization_array, iterations_array, alpha_val_array, userids,
                     urm_train=data.get_urm_train(), urm_test=data.get_urm_test(), filter_already_liked=True,
                     items_to_exclude=[], N=10, verbose=True, write_on_file=True):
        """

        :param factors_array
        :param regularization_array
        :param iterations_array
        :param alpha_val_array
        :param userids: id of the users to take into account during evaluation
        :param urm_train: matrix on which train the model
        :param urm_test: matrix in which test the model
        :param filter_already_liked:
        :param items_to_exclude:
        :param N: evaluate on map@10
        :param verbose:
        :param write_on_file:
        -----------
        :return: _
        """


        #create the initial model
        recommender = AlternatingLeastSquare(urm_train)

        path = 'validation_results/'
        name = 'als'
        folder = time.strftime('%d-%m-%Y')
        filename = '{}/{}/{}{}.csv'.format(path, folder, name, time.strftime('_%H-%M-%S'))
        # create dir if not exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w') as out:
            for f in factors_array:
                for r in regularization_array:
                    for i in iterations_array:
                        for a in alpha_val_array:

                            #train the model with the parameters
                            if verbose:
                                print('\n\nTraining ALS with\n Factors: {}\n Regulatization: {}\n'
                                      'Iterations: {}\n Alpha_val: {}'.format(f, r, i, a))
                                print('\n training phase...')
                            recommender.fit(f, r, i, a)

                            #get the recommendations from the trained model
                            recommendations = recommender.recommend_batch(userids=userids, N=N, filter_already_liked=filter_already_liked,
                                                                          items_to_exclude=items_to_exclude)
                            #evaluate the model with map10
                            map10 = recommender.evaluate(recommendations, test_urm=urm_test)
                            if verbose:
                                print('map@10: {}'.format(map10))

                            #write on external files on folder models_validation
                            if write_on_file:
                                out.write('\n\nFactors: {}\n Regulatization: {}\n Iterations: {}\n '
                                          'Alpha_val: {}\n evaluation map@10: {}'.format(f, r, i, a, map10))


"""
If this file is executed, test the als
"""
if __name__ == '__main__':
    model = AlternatingLeastSquare()
    model.test()
