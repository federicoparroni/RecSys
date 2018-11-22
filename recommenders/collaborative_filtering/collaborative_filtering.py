"""
Collaborative filtering recommender.
"""

from recommenders.distance_based_recommender import DistanceBasedRecommender
import data.data as data
import utils.log as log
from inout.importexport import exportcsv
import time

class CollaborativeFiltering(DistanceBasedRecommender):
    """
    Computes the recommendations for a user by looking for the similar users based on the
    item which they rated
    """

    def __init__(self):
        super(CollaborativeFiltering, self).__init__()
        self.name = 'collaborativefiltering'

    def fit(self, urm_train, k, distance, shrink=0, threshold=0, implicit=True, alpha=None, beta=None, l=None, c=None):
        """
        Initialize the model and compute the similarity matrix S with a distance metric.
        Access the similarity matrix using: self._sim_matrix

        Parameters
        ----------
        urm_train : csr_matrix
            The URM matrix of shape (number_users, number_items) to train the model with.
        k : int
            K nearest neighbour to consider.
        distance : str
            One of the supported distance metrics, check collaborative_filtering_base constants.
        shrink : float, optional
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
        super(CollaborativeFiltering, self).fit(urm_train, k, distance, shrink, threshold, implicit, alpha, beta, l, c)

    # def get_r_hat(self):
    #     """
    #     Return the r_hat matrix as: R^ = R•S ONLY for the TARGET USERS
    #     """
    #     return super(CollaborativeFiltering, self).get_r_hat()

    def run(self, distance, urm_train=None, urm=None, urm_test=None, targetids=None, k=100, shrink=10, threshold=0, implicit=True,
        alpha=None, beta=None, l=None, c=None, with_scores=False, export=True, verbose=True):
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
        _icm = data.get_icm()
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


    def test(self, distance=DistanceBasedRecommender.SIM_SPLUS, k=100, shrink=0, threshold=0, implicit=True, alpha=0.5, beta=0.5, l=0.5, c=0.5):
        """
        Test the model without saving the results. Default distance: SPLUS
        """
        return self.run(distance=distance, k=k, shrink=shrink, threshold=threshold, implicit=implicit, alpha=alpha, beta=beta, l=l, c=c, export=False)


"""
If this file is executed, test the SPLUS distance metric
"""
if __name__ == '__main__':
    model = CollaborativeFiltering()
    model.test()