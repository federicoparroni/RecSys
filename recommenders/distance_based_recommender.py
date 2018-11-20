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
import data

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
        beta: float, optional
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
        # compute and stores the similarity matrix using one of the distance metric
        models={
            #self.SIM_DOTPRODUCT: sim.dot_product(matrix.T, k=k, shrink=shrink, threshold=threshold, binary=implicit),
            self.SIM_COSINE: sim.cosine(matrix.T, k=k, shrink=shrink, threshold=threshold, binary=implicit),
            self.SIM_ASYMCOSINE: sim.asymmetric_cosine(matrix.T, k=k, shrink=shrink, threshold=threshold, binary=implicit, alpha=alpha),
            self.SIM_JACCARD: sim.jaccard(matrix.T, k=k, shrink=shrink, threshold=threshold, binary=implicit),
            self.SIM_DICE: sim.dice(matrix.T, k=k, shrink=shrink, threshold=threshold, binary=implicit),
            self.SIM_TVERSKY: sim.tversky(matrix.T, k=k, shrink=shrink, threshold=threshold, binary=implicit, alpha=alpha, beta=beta),
            self.SIM_P3ALPHA: sim.p3alpha(matrix.T, k=k, shrink=shrink, threshold=threshold, binary=implicit, alpha=alpha),
            self.SIM_RP3BETA: sim.rp3beta(matrix.T, k=k, shrink=shrink, threshold=threshold, binary=implicit, alpha=alpha, beta=beta),
            self.SIM_SPLUS: sim.s_plus(matrix.T, k=k, shrink=shrink, threshold=threshold, binary=implicit, l=l, t1=alpha, t2=beta, c=c)
        }
        self._sim_matrix = models[distance]
    
    def _has_fit(self):
        """
        Check if the model has been fit correctly before being used
        """
        if self._matrix is None or self._sim_matrix is None:
            log.error('Cannot recommend without having fit with a proper matrix. Call method \'fit\'.')
            return False
        else:
            return True

    def recommend(self, userid, N=10, matrix=None, filter_already_liked=True, with_scores=False, items_to_exclude=[]):
        if not self._has_fit():
            return None
        else:
            return self.recommend_batch([userid], N, matrix, filter_already_liked, with_scores, items_to_exclude)

    def recommend_batch(self, userids, N=10, matrix=None, filter_already_liked=True, with_scores=False, items_to_exclude=[], verbose=False):
        if not self._has_fit():
            return None
        else:
            matrix = matrix[[userids]] if matrix is not None else self._matrix[userids]
            # compute the R^ by multiplying R•S
            r_hat = sim.dot_product(matrix, self._sim_matrix, target_rows=None, k=data.N_TRACKS, format_output='csr', verbose=verbose)
            
            if filter_already_liked:
                user_profile_batch = matrix
                r_hat[user_profile_batch.nonzero()] = -np.inf
            if len(items_to_exclude)>0:
                # TO-DO: test this part
                r_hat = r_hat.T
                r_hat[items_to_exclude] = -np.inf
                r_hat = r_hat.T
            
            # convert to np matrix and select only the target rows
            r_hat = r_hat.todense()
            
            # magic code of Mauri 🔮 to take the top N recommendations
            ranking = np.zeros((r_hat.shape[0], N), dtype=np.int)
            
            for i in range(r_hat.shape[0]):
                scores = r_hat[i]      # workaround
                relevant_items_partition = (-scores).argpartition(N)[0,0:N]
                relevant_items_partition_sorting = np.argsort(-scores[0,relevant_items_partition])
                ranking[i] = relevant_items_partition[0,relevant_items_partition_sorting]
            
            # include userids as first column
            recommendations = self._insert_userids_as_first_col(userids, ranking)

            return recommendations
