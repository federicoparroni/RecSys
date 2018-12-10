"""
Content based recommender.
"""

from recommenders.distance_based_recommender import DistanceBasedRecommender
import utils.log as log
import numpy as np
import similaripy as sim
import data.data as data
from inout.importexport import exportcsv
import time
import data.data as d
import sys
import scipy.sparse as sps

class ContentBasedRecommender(DistanceBasedRecommender):
    """
    Computes the recommendations for a user by looking for the most similar items that he
    has already interacted with.
    """

    def __init__(self):
        super(ContentBasedRecommender, self).__init__()
        self.name = 'content_based'

    def fit(self, urm, icm, k=0, distance='cosine', shrink=0, threshold=0, alpha=None, beta=None, l=None, c=None):
        """
        Initialize the model and compute the Similarity_MFD matrix S with a distance metric.
        Access the Similarity_MFD matrix using: self._sim_matrix

        Parameters
        ----------
        icm : csr_matrix
            The ICM matrix of shape (number_items, number_features).
        k : int
            K nearest neighbour to consider.
        distance : str
            One of the supported distance metrics, check collaborative_filtering_base constants.
        shrink : float, optional
            Shrink term used in the normalization
        alpha: float, optional, included in [0,1]
        beta: float, optional, included in [0,1]
        l: float, optional, balance coefficient used in s_plus distance, included in [0,1]
        c: float, optional, cosine coefficient, included in [0,1]
        """
        self.urm = urm
        return super(ContentBasedRecommender, self).fit(matrix=icm, k=k, distance=distance, shrink=shrink, threshold=threshold, implicit=False, alpha=alpha, beta=beta, l=l, c=c)

    def recommend(self, userid, urm=None, N=10, filter_already_liked=True, with_scores=False, items_to_exclude=[]):
        if not userid >= 0:
            log.error('Invalid user id')
            return None
        return self.recommend_batch([userid], urm, N, filter_already_liked, with_scores, items_to_exclude)

    def recommend_batch(self, userids, urm=None, N=10, filter_already_liked=True, with_scores=False, items_to_exclude=[],
                        verbose=False):
        if not self._has_fit():
            return None

        if userids is not None:
            if len(userids) > 0:
                matrix = urm[userids] if urm is not None else data.get_urm()[userids]
            else:
                return []
        else:
            print('Recommending for all users...')
            matrix = urm if urm is not None else data.get_urm()

        # compute the R^ by multiplying R•S
        self.r_hat = matrix * self._sim_matrix
        
        if filter_already_liked:
            user_profile_batch = matrix
            self.r_hat[user_profile_batch.nonzero()] = -np.inf
        if len(items_to_exclude)>0:
            # TO-DO: test this part because it does not work!
            self.r_hat = self.r_hat.T
            self.r_hat[items_to_exclude] = -np.inf
            self.r_hat = self.r_hat.T
        
        recommendations = self._extract_top_items(self.r_hat, N=N)
        return self._insert_userids_as_first_col(userids, recommendations).tolist()

    def _print(self, distance, k, shrink, threshold, alpha, beta, l, c):
        print('{}; distance: {}, KNN: {}, shrink: {}, threshold: {}, alpha: {}, beta: {}, l: {}, c: {}.'.format(
            self.name, distance, k, shrink, threshold, alpha, beta, l, c))

    def run(self, distance, urm=None, icm=None, urm_test=None, targetids=None, k=100, shrink=10, threshold=0,
        alpha=None, beta=None, l=None, c=None, with_scores=False, export=True, verbose=True):
        """
        Run the model and export the results to a file

        Parameters
        ----------
        distance : str, distance metric
        urm : csr matrix, URM. If None, used: data.get_urm_train(). This should be the
            entire URM for which the targetids corresponds to the row indexes.
        icm : csr matrix, ICM. If None, used: data.get_icm()
        urm_test : csr matrix, urm where to test the model. If None, use: data.get_urm_test()
        targetids : list, target user ids. If None, use: data.get_target_playlists()
        k : int, K nearest neighbour to consider
        shrink : float, shrink term used in the normalization
        threshold : float, all the values under this value are cutted from the final result

        Returns
        -------
        recs: (list) recommendations
        map10: (float) MAP10 for the provided recommendations
        """
        # _urm = data.get_urm_train()
        # _icm = data.get_icm()
        # _urm_test = data.get_urm_test()
        # _targetids = data.get_target_playlists()
        #_targetids = data.get_all_playlists()

        start = time.time()

        # urm = _urm if urm is None else urm
        # icm = _icm if icm is None else icm
        # urm_test = _urm_test if urm_test is None else urm_test
        # targetids = _targetids if targetids is None else targetids

        self._print(distance, k, shrink, threshold, alpha, beta, l, c)

        self.fit(urm, icm=icm, k=k, distance=distance, shrink=shrink, threshold=threshold, alpha=alpha, beta=beta, l=l, c=c)
        recs = self.recommend_batch(userids=targetids, urm=urm, N=10, verbose=verbose)

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


    #def test(self, distance=DistanceBasedRecommender.SIM_SPLUS, k=100, shrink=0, threshold=0, alpha=0.5, beta=0.5, l=0.5, c=0.5):
    def test(self, distance=DistanceBasedRecommender.SIM_SPLUS, k=600, shrink=10, threshold=0, alpha=0.25, beta=0.5, l=0.5, c=0.25):
        """
        meant as a shortcut to run the model after the validation procedure,
        allowing the export of the scores on the playlists or of the estimated csr matrix
        """
        recs, map = self.run(urm=d.get_urm(),
                             icm=d.get_icm(),
                             targetids=d.get_target_playlists(),
                             distance=distance,
                             k=k, shrink=shrink,
                             threshold=threshold,
                             alpha=alpha,
                             beta=beta,
                             l=l,
                             c=c,
                             export=export_results)

        if export_r_hat:
            print('saving estimated urm')
            self.save_r_hat()
        return recs, map

    def validate(self, urm, icm, urm_test,
                 targetids,distance=[DistanceBasedRecommender.SIM_SPLUS], k=[100], shrink=[0],
                 threshold=[0], alpha=[0.5],
                 beta=[0.5], l=[0.5], c=[0.5], log_path=None):

        if log_path != None:
            orig_stdout = sys.stdout
            f = open(log_path + '/' + self.name + ' ' + time.strftime('_%H-%M-%S') + ' ' +
                     time.strftime('%d-%m-%Y') + '.txt', 'w')
            sys.stdout = f

        for dist in distance:
            for k_ in k:
                for s in shrink:
                    for t in threshold:
                        for a in alpha:
                            for b in beta:
                                for l_ in l:
                                    for c_ in c:
                                        self.run(dist,
                                                 urm=urm,
                                                 icm=icm,
                                                 urm_test=urm_test,
                                                 targetids=targetids,
                                                 k=k_,
                                                 shrink=s,
                                                 threshold=t,
                                                 alpha=a,
                                                 beta=b,
                                                 l=l_,
                                                 c=c_,
                                                 with_scores=False,
                                                 export=False,
                                                 verbose=True)

        if log_path != None:
            sys.stdout = orig_stdout
            f.close()
    #
    # def get_r_hat(self):
    #     r_hat = sim.dot_product(self.urm, self._sim_matrix, target_rows=data.get_target_playlists(),
    #                             k=data.N_TRACKS, format_output='csr')
    #     return r_hat[data.get_target_playlists()]

"""
If this file is executed, test the SPLUS distance metric
"""
if __name__ == '__main__':
    print()
    log.success('++ What do you want to do? ++')
    log.warning('(t) Test the model with some default params')
    log.warning('(r) Save the R^')
    log.warning('(s) Save the similarity matrix')
    #log.warning('(v) Validate the model')
    log.warning('(x) Exit')
    arg = input()[0]
    print()
    
    model = ContentBasedRecommender()
    if arg == 't':
        # recs = model.recommend_batch(userids=data.get_target_playlists(), urm=data.get_urm_train())
        # model.evaluate(recommendations=recs, test_urm=data.get_urm_test())
        model.test(distance=model.SIM_SPLUS, k=500,alpha=0.75,beta=1,shrink=500,l=0.5,c=0.5)
    elif arg == 'r':
        log.info('Wanna save for evaluation (y/n)?')
        choice = input()[0] == 'y'
        model.fit(urm=data.get_urm_train_2(),icm=data.get_icm(), distance=model.SIM_SPLUS,k=500,shrink=500,alpha=0.75,beta=1,l=0.5,c=0.5)
        print('Saving the R^...')
        model.save_r_hat(evaluation=choice)
    elif arg == 's':
        model.fit(urm=data.get_urm_train(),icm=data.get_icm(), distance=model.SIM_SPLUS,k=500,shrink=500,alpha=0.75,beta=1,l=0.5,c=0.5)
        print('Saving the similarity matrix...')
        sps.save_npz('raw_data/saved_sim_matrix_evaluation/{}'.format(model.name), model.get_sim_matrix())
    # elif arg == 'v':
    #     model.validate(....)
    elif arg == 'x':
        pass
    else:
        log.error('Wrong option!')

    # recs = model.recommend_batch(userids=data.get_target_playlists(), urm=data.get_urm_train())
    # recs_seq = model.recommend_batch(userids=data.get_sequential_target_playlists(), urm=data.get_urm_train())
    # model.evaluate(recommendations=recs, test_urm=data.get_urm_test())
    # model.evaluate(recommendations=recs_seq, test_urm=data.get_urm_test())
