"""
Base class for a distance based recommender.
Supports several distance metrics, thanks to similaripy library.
See https://github.com/bogliosimone/similaripy/blob/master/guide/temp_guide.md
for documentation and distance formulas
"""

from recommenders.recommender_base import RecommenderBase
import utils.log as log
import numpy as np
import similaripy as sim
import data.data as data

class DistanceBasedRecommender(RecommenderBase):
    """
    Base class for a distance based recommender.
    Supports several distance metrics, thanks to similaripy library
    """

    #SIM_DOTPRODUCT = 'dotproduct'
    SIM_COSINE = 'cosine'
    SIM_ASYMCOSINE = 'asymcosine'
    SIM_JACCARD = 'jaccard'
    SIM_DICE = 'dice'
    SIM_TVERSKY = 'tversky'

    SIM_P3ALPHA = 'p3alpha'
    SIM_RP3BETA = 'rp3beta'

    SIM_SPLUS = 'splus'

    def __init__(self):
        super(DistanceBasedRecommender, self).__init__()
        self.name = 'distancebased'
        self._sim_matrix = None
        self._matrix = None

    def fit(self, matrix, k, distance, shrink=0, threshold=0, implicit=True, alpha=None, beta=None, l=None, c=None):
        """
        Initialize the model and compute the similarity matrix S with a distance metric.
        Access the similarity matrix using: self._sim_matrix

        Parameters
        ----------
        matrix : csr_matrix
            A sparse matrix. For example, it can be the URM of shape (number_users, number_items).
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
        alpha = -1 if alpha is None else alpha
        beta = -1 if beta is None else beta
        l = -1 if l is None else l
        c = -1 if c is None else c
        if distance==self.SIM_ASYMCOSINE and not(0 <= alpha <= 1):
            log.error('Invalid parameter alpha in asymmetric cosine similarity!')
            return
        if distance==self.SIM_TVERSKY and not(0 <= alpha <= 1 and 0 <= beta <= 1):
            log.error('Invalid parameter alpha/beta in tversky similarity!')
            return
        if distance==self.SIM_P3ALPHA and alpha is None:
            log.error('Invalid parameter alpha in p3alpha similarity')
            return
        if distance==self.SIM_RP3BETA and alpha is None and beta is None:
            log.error('Invalid parameter alpha/beta in rp3beta similarity')
            return
        if distance==self.SIM_SPLUS and not(0 <= l <= 1 and 0 <= c <= 1 and 0 <= alpha <= 1 and 0 <= beta <= 1):
            log.error('Invalid parameter alpha/beta/l/c in s_plus similarity')
            return
        # save the urm for later usage
        self._matrix = matrix
        # compute and stores the similarity matrix using one of the distance metric: S = R'•R
        if distance==self.SIM_COSINE:
            self._sim_matrix = sim.cosine(matrix.T, k=k, shrink=shrink, threshold=threshold, binary=implicit)
        elif distance==self.SIM_ASYMCOSINE:
            self._sim_matrix = sim.asymmetric_cosine(matrix.T, k=k, shrink=shrink, threshold=threshold, binary=implicit, alpha=alpha)
        elif distance==self.SIM_JACCARD:
            self._sim_matrix = sim.jaccard(matrix.T, k=k, shrink=shrink, threshold=threshold, binary=implicit)
        elif distance==self.SIM_DICE:
            self._sim_matrix = sim.dice(matrix.T, k=k, shrink=shrink, threshold=threshold, binary=implicit)
        elif distance==self.SIM_TVERSKY:
            self._sim_matrix = sim.tversky(matrix.T, k=k, shrink=shrink, threshold=threshold, binary=implicit, alpha=alpha, beta=beta)
        elif distance==self.SIM_P3ALPHA:
            self._sim_matrix = sim.p3alpha(matrix.T, k=k, shrink=shrink, threshold=threshold, binary=implicit, alpha=alpha)
        elif distance==self.SIM_RP3BETA:
            self._sim_matrix = sim.rp3beta(matrix.T, k=k, shrink=shrink, threshold=threshold, binary=implicit, alpha=alpha, beta=beta)
        elif distance==self.SIM_SPLUS:
            self._sim_matrix = sim.s_plus(matrix.T, k=k, shrink=shrink, threshold=threshold, binary=implicit, l=l, t1=alpha, t2=beta, c=c)
        else:
            log.error('Invalid distance metric: {}'.format(distance))
        #self.SIM_DOTPRODUCT: sim.dot_product(matrix.T, k=k, shrink=shrink, threshold=threshold, binary=implicit)
    
    def _has_fit(self):
        """
        Check if the model has been fit correctly before being used
        """
        if self._matrix is None or self._sim_matrix is None:
            log.error('Cannot recommend without having fit with a proper matrix. Call method \'fit\'.')
            return False
        else:
            return True

    def get_r_hat(self):
        """
        Return the r_hat matrix as: R^ = R•S ONLY for the TARGET USERS
        """
        _urm = self._matrix[data.get_target_playlists()]
        return sim.dot_product(_urm, self._sim_matrix, target_rows=None, k=data.N_PLAYLISTS, format_output='csr', verbose=False)

    def recommend(self, userid, N=10, urm=None, filter_already_liked=True, with_scores=False, items_to_exclude=[]):
        if not self._has_fit():
            return None
        
        return self.recommend_batch([userid], N, filter_already_liked, with_scores, items_to_exclude)

    def recommend_batch(self, userids, N=10, urm=None, filter_already_liked=True, with_scores=False, items_to_exclude=[], verbose=False):
        if not self._has_fit():
            return None

        if userids is not None:
            if len(userids) > 0:
                matrix = urm[userids] if urm is not None else self._matrix[userids]
            else:
                return []
        else:
            print('Recommending for all users...')
            matrix = urm if urm is not None else self._matrix
        
        # compute the R^ by multiplying R•S
        self.r_hat = sim.dot_product(matrix, self._sim_matrix, target_rows=None, k=data.N_PLAYLISTS, format_output='csr', verbose=verbose)
        
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

    def _extract_top_items(self, r_hat, N):
        # convert to np matrix
        r_hat = r_hat.todense()
        
        # magic code of Mauri 🔮 to take the top N recommendations
        ranking = np.zeros((r_hat.shape[0], N), dtype=np.int)
        for i in range(r_hat.shape[0]):
            scores = r_hat[i]
            relevant_items_partition = (-scores).argpartition(N)[0,0:N]
            relevant_items_partition_sorting = np.argsort(-scores[0,relevant_items_partition])
            ranking[i] = relevant_items_partition[0,relevant_items_partition_sorting]
        
        # include userids as first column
        return ranking
