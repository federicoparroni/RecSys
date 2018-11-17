"""
Base class for a collaborative filtering recommender.
Supports several distance metrics, thanks to similaripy library.
See https://github.com/bogliosimone/similaripy/blob/master/guide/temp_guide.md
for documentation and distance formulas
"""

from recommenders.recommender_base import RecommenderBase
import utils.log as log
import similaripy as sim

SIM_DOTPRODUCT = 'dotproduct'
SIM_COSINE = 'cosine'
SIM_ASYMCOSINE = 'asymcosine'
SIM_JACCARD = 'jaccard'
SIM_DICE = 'dice'
SIM_TVERSKY = 'tversky'

SIM_P3ALPHA = 'p3alpha'
SIM_RP3BETA = 'rp3beta'

SIM_SPLUS = 'splus'

class CollaborativeFilteringBase(RecommenderBase):
    """
    Base class for a collaborative filtering recommender.
    Supports several distance metrics, thanks to similaripy library
    """

    def __init__(self):
        self._sim_matrix = None

    def fit(self, urm, k, distance, shrink=0, threshold=0, implicit=True, alpha=None, beta=None, l=None, c=None):
        """
        Initialize the model with a distance metric

        Parameters
        ----------
        urm : csr_matrix
            A sparse matrix of shape (number_users, number_items).
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
        beta: float, optional
        l: float, optional, balance coefficient used in s_plus distance, included in [0,1]
        c: float, optional, cosine coefficient, included in [0,1]
        """
        if distance==SIM_ASYMCOSINE and 0 <= alpha <= 1:
            log.error('Invalid parameter alpha in asymmetric cosine similarity!')
            return
        if distance==SIM_TVERSKY and 0 <= alpha <= 1 and 0 <= beta <= 1
            log.error('Invalid parameter alpha/beta in tversky similarity!')
            return
        if distance==SIM_P3ALPHA and alpha is not None
            log.error('Invalid parameter alpha in p3alpha similarity')
            return
        if distance==SIM_RP3BETA and alpha is not None and beta is not None
            log.error('Invalid parameter alpha/beta in rp3beta similarity')
            return
        if distance==SIM_SPLUS and 0 <= l <= 1 and 0 <= c <= 1 and 0 <= alpha <= 1 and 0 <= beta <= 1:
            log.error('Invalid parameter alpha/beta/l/c in s_plus similarity')
            return

        models={
            SIM_DOTPRODUCT: sim.dot_product(urm.T, k=k, shrink=shrink, threshold=threshold, binary=implicit),
            SIM_COSINE: sim.cosine(urm.T, k=k, shrink=shrink, threshold=threshold, binary=implicit),
            SIM_ASYMCOSINE: sim.asymmetric_cosine(urm.T, k=k, shrink=shrink, threshold=threshold, binary=implicit, alpha=alpha),
            SIM_JACCARD: sim.jaccard(urm.T, k=k, shrink=shrink, threshold=threshold, binary=implicit),
            SIM_DICE: sim.dice(urm.T, k=k, shrink=shrink, threshold=threshold, binary=implicit),
            SIM_TVERSKY: sim.tversky(urm.T, k=k, shrink=shrink, threshold=threshold, binary=implicit),
            SIM_P3ALPHA: sim.p3alpha(urm.T, k=k, shrink=shrink, threshold=threshold, binary=implicit, alpha=alpha),
            SIM_RP3BETA: sim.rp3beta(urm.T, k=k, shrink=shrink, threshold=threshold, binary=implicit, alpha=alpha, beta=beta),
            SIM_SPLUS: sim.s_plus(urm.T, k=k, shrink=shrink, threshold=threshold, binary=implicit, l=l, t1=alpha, t2=beta, c=c)
        }
        self._sim_matrix = models[distance]
    
    def _has_fit(self):
        """
        Check if the model has been fit correctly before being used
        """
        if self._sim_matrix is None:
            log.error('Cannot recommend without having fit with a URM. Call method \'fit\'.')
            return False
        else:
            return True

    def recommend(self, userid, N=10, urm=None, filter_already_liked=True, with_scores=True, items_to_exclude=[]):
        if not self._has_fit():
            return None
        else:
            recs = sim.dot_product(urm, self._sim_matrix, target_rows=[userid], k=N)
            return recs

    def recommend_batch(self, userids, N=10, urm=None, filter_already_liked=True, with_scores=True, items_to_exclude=[], verbose=False):
        if not self._has_fit():
            return None
        else:
            recs = sim.dot_product(urm, self._sim_matrix, target_rows=[userids], k=N)
            return recs