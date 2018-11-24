from sklearn.utils.extmath import randomized_svd
from recommenders.recommender_base import RecommenderBase
import data.data as data
import scipy.sparse as sps
import numpy as np
import utils.log as log
from inout.importexport import exportcsv
import time
import os


class Pure_SVD(RecommenderBase):

    def __init__(self):
        self.name = 'pureSVD'

    def fit(self, urm_train, num_factors=50, iteration='auto'):
        self.urm = urm_train
        U, Sigma, VT = randomized_svd(self.urm,
                                      n_components=num_factors,
                                      random_state=None,
                                      n_iter=iteration)
        self.s_Vt = sps.diags(Sigma)*VT
        self.U = U


    def get_r_hat(self, load_from_file=False, path=''):
        """
        :param load_from_file: if the matrix has been saved can be set to true for load it from it
        :param path: path in which the matrix has been saved
        -------
        :return the extimated urm from the recommender
        """
        U_filtered = self.U[data.get_target_playlists()]
        r_hat = U_filtered.dot(self.s_Vt)
        return sps.csr_matrix(r_hat)

    def recommend(self, userid, N=10, urm=None, filter_already_liked=True, with_scores=False, items_to_exclude=[]):
        """
        Recommend the N best items for the specified user

        Parameters
        ----------
        userid : int
            The user id to calculate recommendations for
        urm : csr_matrix
            A sparse matrix of shape (number_users, number_items). This allows to look
            up the liked items and their weights for the user. It is used to filter out
            items that have already been liked from the output, and to also potentially
            giving more information to choose the best items for this user.
        N : int, optional
            The number of recommendations to return
        items_to_exclude : list of ints, optional
            List of extra item ids to filter out from the output

        Returns
        -------
        list
            List of length N of (itemid, score) tuples: [ (18,0.7), (51,0.5), ... ]
        """
        pass

    def recommend_batch(self, userids,  N=10, filter_already_liked=True, with_scores=False, items_to_exclude=[]):
        user_profile = self.urm[userids]

        U_filtered = self.U[userids]
        scores = U_filtered.dot(self.s_Vt)

        if filter_already_liked:
            scores[user_profile.nonzero()] = -np.inf

        ranking = np.zeros((scores.shape[0], N), dtype=np.int)
        #scores = scores.todense()

        for row_index in range(scores.shape[0]):
            scores_row = scores[row_index]

            relevant_items_partition = (-scores_row).argpartition(N)[0:N]
            relevant_items_partition_sorting = np.argsort(-scores_row[relevant_items_partition])
            ranking[row_index] = relevant_items_partition[relevant_items_partition_sorting[0:N]]

        recommendations = self._insert_userids_as_first_col(userids, ranking)
        return recommendations

    def run(self, num_factors, urm_train=None, urm=None, urm_test=None, targetids=None, with_scores=False, export=True, verbose=True):
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

        self.fit(urm_train=urm_train, num_factors=num_factors)
        recs = self.recommend_batch(userids=targetids)

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
        return self.run(num_factors=num_factors, export=False)

    def validate(self, factors_array, iteration_array, urm_train=data.get_urm_train(), urm_test=data.get_urm_test(), verbose=True,
                 write_on_file=True, userids=data.get_target_playlists(), N=10, filter_already_liked=True, items_to_exclude=[]):

        #create the initial model
        recommender = Pure_SVD()

        path = 'validation_results/'
        name = 'pure_SVD'
        folder = time.strftime('%d-%m-%Y')
        filename = '{}/{}/{}{}.csv'.format(path, folder, name, time.strftime('_%H-%M-%S'))
        # create dir if not exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w') as out:
            for f in factors_array:
                for i in iteration_array:
                    #train the model with the parameters
                    if verbose:
                        print('\n\nTraining PURE_SVD with\n Factors: {}\n Iteration: {}\n'.format(f, i))
                        print('\n training phase...')
                    recommender.fit(urm_train=urm_train, num_factors=f, iteration=i)

                    #get the recommendations from the trained model
                    recommendations = recommender.recommend_batch(userids=userids, N=N, filter_already_liked=filter_already_liked,
                                                                  items_to_exclude=items_to_exclude)
                    #evaluate the model with map10
                    map10 = recommender.evaluate(recommendations, test_urm=urm_test)
                    if verbose:
                        print('map@10: {}'.format(map10))

                    #write on external files on folder models_validation
                    if write_on_file:
                        out.write('\n\nFactors: {}\n Iteration: {}\n evaluation map@10: {}'.format(f, i, map10))



"""
If this file is executed, test the SPLUS distance metric
"""
if __name__ == '__main__':
    model = Pure_SVD()
    model.validate(factors_array=[600, 640, 700, 730, 800, 860, 1000], iteration_array=[1, 2, 5])