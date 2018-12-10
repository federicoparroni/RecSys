"""
Collaborative filtering recommender.
"""

from recommenders.distance_based_recommender import DistanceBasedRecommender
import data.data as data
import utils.log as log
import similaripy as sim
import numpy as np
from inout.importexport import exportcsv
import time
import scipy.sparse as sps

class CFUserBased(DistanceBasedRecommender):
    """
    Computes the recommendations for a user by looking for the similar users based on the
    item which they rated
    """

    def __init__(self):
        super(CFUserBased, self).__init__()
        self._matrix_mul_order = 'inverse'
        self.name = 'CFuser'

    def fit(self, urm_train, k, distance, shrink=0, threshold=0, implicit=True, alpha=None, beta=None, l=None, c=None, verbose=False):
        """
        Initialize the model and compute the Similarity_MFD matrix S with a distance metric.
        Access the Similarity_MFD matrix using: self._sim_matrix

        Parameters
        ----------
        urm_train: csr_matrix
            The URM matrix of shape (number_users, number_items) to train the model with.
        k: int
            K nearest neighbour to consider.
        distance: str
            One of the supported distance metrics, check collaborative_filtering_base constants.
        shrink: float, optional
            Shrink term used in the normalization
        threshold: float, optional
            All the values under this value are cutted from the final result
        implicit: bool, optional
            If true, treat the URM as implicit, otherwise consider explicit ratings (real values) in the URM
        alpha: float, optional, included in [0,1]
        beta: float, optional, included in [0,1]
        l: float, optional, balance coefficient used in s_plus distance, included in [0,1]
        c: float, optional, cosine coefficient, included in [0,1]
        """
        self.urm = urm_train
        return super(CFUserBased, self).fit(urm_train, k=k, distance=distance, shrink=shrink, threshold=threshold,
                                            implicit=implicit, alpha=alpha, beta=beta, l=l, c=c, verbose=verbose)

    def get_r_hat(self, only_target=True, verbose=False):
        """
        Return the r_hat matrix as: R^ = S•R, ONLY for the TARGET USERS
        """
        return super(CFUserBased, self).get_r_hat(verbose=verbose)


    def run(self, distance, urm_train=None, urm=None, urm_test=None, targetids=None, k=100, shrink=10, threshold=0,
            implicit=True, alpha=None, beta=None, l=None, c=None, with_scores=False, export=True, verbose=True):
        """
        Run the model and export the results to a file

        Parameters
        ----------
        distance : str, distance metric
        urm : csr matrix, URM. If None, used: data.get_urm_train(). This should be the
            entire URM for which the targetids corresponds to the row indexes.
        urm_test : csr matrix, urm where to test the model. If None, use: data.get_urm_test()
        targetids : list, target user ids. If None, use: data.get_target_playlists()
        k : int, K nearest neighbour to consider
        shrink : float, shrink term used in the normalization
        threshold : float, all the values under this value are cutted from the final result
        implicit : bool, if true, treat the URM as implicit, otherwise consider explicit ratings (real values) in the URM

        Returns
        -------
        recs: (list) recommendations
        map10: (float) MAP10 for the provided recommendations
        """
        _urm = data.get_urm_train()
        _urm_test = data.get_urm_test()
        _targetids = data.get_target_playlists()
        #_targetids = data.get_all_playlists()

        start = time.time()

        urm_train = _urm if urm_train is None else urm_train
        urm = _urm if urm is None else urm
        urm_test = _urm_test if urm_test is None else urm_test
        targetids = _targetids if targetids is None else targetids

        self.fit(urm_train, k=k, distance=distance, alpha=alpha, beta=beta, c=c, l=l, shrink=shrink, threshold=threshold, implicit=implicit)
        recs = self.recommend_batch(targetids, urm=urm, with_scores=with_scores, verbose=verbose)

        map10 = None
        if len(recs) > 0:
            map10 = self.evaluate(recs, test_urm=urm_test, verbose=verbose)
        else:
            log.warning('No recommendations available, skip evaluation')

        if export:
            exportcsv(recs, path='submission', name='{}_{}'.format(self.name,distance), verbose=verbose)

        if verbose:
            log.info('Run in: {:.2f}s'.format(time.time()-start))
        
        return recs, map10


    def test(self, distance=DistanceBasedRecommender.SIM_SPLUS, k=200, shrink=0, threshold=0, implicit=True, alpha=0.5, beta=0.5, l=0.5, c=0.5):
        """
        Test the model without saving the results. Default distance: SPLUS
        """
        return self.run(distance=distance, k=k, shrink=shrink, threshold=threshold, implicit=implicit, alpha=alpha, beta=beta, l=l, c=c, export=False)


"""
If this file is executed, test the SPLUS distance metric
"""
if __name__ == '__main__':
    print()
    log.success('++ What do you want to do? ++ \t\t\t\t\t e')
    log.warning('(t) Test the model with some default params')
    log.warning('(r) Save the R^')
    log.warning('(s) Save the similarity matrix')
    #log.warning('(v) Validate the model')
    log.warning('(x) Exit')
    arg = input()[0]
    print()
    
    model = CFUserBased()
    if arg == 't':
        # recs = model.recommend_batch(userids=data.get_target_playlists(), urm=data.get_urm_train())
        # model.evaluate(recommendations=recs, test_urm=data.get_urm_test())
        model.test(distance=CFUserBased.SIM_SPLUS, k=600,alpha=0.25,beta=0.5,shrink=10,l=0.25,c=0.5)
    elif arg == 'r':
        log.info('Wanna save for evaluation (y/n)?')
        choice = input()[0] == 'y'
        model.fit(data.get_urm_train(), distance=model.SIM_SPLUS,k=400,alpha=0.25,beta=0.5,shrink=0,l=0.25,c=0.25)
        print('Saving the R^...')
        model.save_r_hat(evaluation=choice)
    elif arg == 's':
        model.fit(data.get_urm_train(), distance=model.SIM_SPLUS,k=400,alpha=0.25,beta=0.5,shrink=0,l=0.25,c=0.25)
        print('Saving the similarity matrix...')
        sps.save_npz('raw_data/saved_sim_matrix_evaluation/{}'.format(model.name), model.get_sim_matrix())
    # elif arg == 'v':
    #     model.validate(iterations=50, urm_train=data.get_urm_train(), urm_test=data.get_urm_test(), targetids=data.get_target_playlists(),
    #         distance=model.SIM_P3ALPHA, k=(100, 600), alpha=(0,1), beta=(0, 1),shrink=(0,100),l=(0,1),c=(0,1))
    elif arg == 'e':
        print('Grazie Edo...')
    elif arg == 'x':
        pass
    else:
        log.error('Wrong option!')


    # rec = model.recommend_batch(userids=data.get_target_playlists(), urm=data.get_urm_train())
    # rec_seq = model.recommend_batch(userids=data.get_sequential_target_playlists(), urm=data.get_urm_train())
    # rec_non_seq = model.recommend_batch(userids=data.get_all_playlists()[::2113], urm=data.get_urm_train())
    # model.evaluate(recommendations=rec, test_urm=data.get_urm_test())
    # model.evaluate(recommendations=rec_seq, test_urm=data.get_urm_test())
    # model.evaluate(recommendations=rec_non_seq, test_urm=data.get_urm_test())
