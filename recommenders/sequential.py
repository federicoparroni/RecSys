"""
Sequential recommender.

http://ls13-www.cs.tu-dortmund.de/homepage/publications/jannach/Conference_RecSys_2015_pl.pdf
"""

from recommenders.distance_based_recommender import DistanceBasedRecommender
import data.data as data
import preprocessing.sequential as ps
import utils.log as log
import cythoncompiled.seq_similarity as seqsim
import similaripy as sim
import numpy as np
from inout.importexport import exportcsv
import time
import utils.dated_directory as datedir
import scipy.sparse as sps

class SequentialRecommender(DistanceBasedRecommender):
    """
    Computes the recommendations for a playlsit by looking for the similar sequences in all the dataset
    """

    def __init__(self, h, split_perc):
        """
        h: (int), length of the sequences
        split_perc: (float) validation split percentage, 0 to skip the creation of the validation set
        """
        super(SequentialRecommender, self).__init__()
        self.name = 'sequential'
        self.h = h
        
        # build sequences dataset and cache it
        self.urm_train, self.urm_test, self.sequences, self.target_indices = ps.preprocess(h=h, split_perc=split_perc)
        target_ids = data.get_target_playlists()[0:data.NUM_SEQUENTIAL]
        self.target_ids = np.array(target_ids)
        self.already_liked_indices = (self.urm_train[target_ids]).nonzero()
        self.H = seqsim.getH(self.sequences)

    def fit(self, k, distance, shrink=0, alpha=None, beta=None, l=None, c=None, verbose=False):
        """
        Initialize the model and compute the similarity matrix S with a distance metric.
        Access the similarity matrix using: self._sim_matrix

        Parameters
        ----------
        k: (int), K nearest neighbour to consider.
        distance: (str)
            One of the supported distance metrics, check collaborative_filtering_base constants.
        shrink: (float), optional
            Shrink term used in the normalization
        threshold: (float), optional
            All the values under this value are cut from the final result
        implicit: (bool), optional
            If true, treat the URM as implicit, otherwise consider explicit ratings (real values) in the URM
        alpha: (float), optional, included in [0,1]
        beta: (float), optional, included in [0,1]
        l: (float), optional, balance coefficient used in s_plus distance, included in [0,1]
        c: (float), optional, cosine coefficient, included in [0,1]
        """
        super(SequentialRecommender, self).fit(self.H, k=k, distance=distance, shrink=shrink,
                                            implicit=True, alpha=alpha, beta=beta, l=l, c=c, verbose=verbose)
        self._sim_matrix = self._sim_matrix / self.h
        return self._sim_matrix

    def get_r_hat(self, verbose=False):
        """
        Return the R^ matrix as: R^ = S•H, ONLY for the target playlists last sequences
        """
        return sim.dot_product(self._sim_matrix, self.H, target_rows=self.target_indices,
                                k=self.H.shape[0], format_output='csr', verbose=verbose)[self.target_indices]
    
    def recommend_batch(self, N=10, filter_already_liked=True, items_to_exclude=[], verbose=False):
        if not self._has_fit():
            return None
            
        # R^ = S•H
        R_hat = self.get_r_hat(verbose=verbose)
        #R_hat = (self._sim_matrix.tocsr())[self.target_indices].dot(self.H)
        
        if filter_already_liked:
            # remove from the R^ the items already in the R
            R_hat[self.already_liked_indices] = -np.inf
        if len(items_to_exclude)>0:
            # TO-DO: test this part because it does not work!
            R_hat = R_hat.T
            R_hat[items_to_exclude] = -np.inf
            R_hat = R_hat.T

        recommendations = self._extract_top_items(R_hat, N=N)
        return self._insert_userids_as_first_col(self.target_ids, recommendations).tolist()


    def run(self, distance, h, k=100, shrink=10, threshold=0,
            alpha=None, beta=None, l=None, c=None, export=True, verbose=True):
        """
        Run the model and export the results to a file

        Parameters
        ----------
        distance : str, distance metric
        k : int, K nearest neighbour to consider
        shrink : float, shrink term used in the normalization
        threshold : float, all the values under this value are cutted from the final result
        implicit : bool, if true, treat the URM as implicit, otherwise consider explicit ratings (real values) in the URM

        Returns
        -------
        recs: (list) recommendations
        map10: (float) MAP10 for the provided recommendations
        """
        start = time.time()

        self.fit(k=k, distance=distance, shrink=shrink, alpha=alpha, beta=beta, l=l, c=c, verbose=verbose)
        recs = self.recommend_batch(N=10, filter_already_liked=True, verbose=verbose)

        map10 = None
        if len(recs) > 0:
            map10 = self.evaluate(recs, test_urm=self.urm_test, verbose=verbose)
        else:
            log.warning('No recommendations available, skip evaluation')

        if export:
            exportcsv(recs, path='submission', name='{}_{}'.format(self.name,distance), verbose=verbose)

        if verbose:
            log.info('Run in: {:.2f}s'.format(time.time()-start))
        
        return recs, map10


    def test(self, distance=DistanceBasedRecommender.SIM_SPLUS, h=5, k=200, shrink=0, threshold=0, alpha=0.5, beta=0.5, l=0.5, c=0.5):
        """
        Test the model without saving the results. Default distance: SPLUS
        """
        return self.run(distance=distance, h=h, k=k, shrink=shrink, threshold=threshold,
                        alpha=alpha, beta=beta, l=l, c=c, export=False)


def validate(self, ks, alphas, betas, ls, cs, shrinks, filename='sequential_validation', path='validation_results', verbose=False):
    distance = SequentialRecommender.SIM_COSINE

    # ks = [100, 200, 300]
    # alphas = [0.25, 0.5, 0.75]
    # betas = [0.25, 0.5, 0.75]
    # ls = [0.25, 0.5, 0.75]
    # cs = [0.25, 0.5, 0.75]
    # shrinks = [0, 10, 30]
    """
    i=0
    tot=len(ks)*len(alphas)*len(betas)*len(ls)*len(cs)*len(shrinks)

    filename = datedir.create_folder(rootpath=path, filename=filename, extension='txt')
    with open(filename, 'w') as file:
        for k in ks:
            for a in alphas:
                for b in betas:
                    for l in ls:
                        for c in cs:
                            for shrink in shrinks:
                                model = CFItemBased()
                                recs, map10 = model.run(distance=distance, k=k, shrink=shrink, alpha=a, beta=b, c=c, l=l, export=False, verbose=verbose)
                                logmsg = 'MAP: {} \tknn: {} \ta: {} \tb: {} \tl: {} \tc: {} \tshrink: {}\n'.format(map10,k,a,b,l,c,shrink)
                                #log.warning(logmsg)
                                file.write(logmsg)
                                
                                i+=1
                                log.progressbar(i,tot, prefix='Validation: ')
    """

"""
If this file is executed, test the SPLUS distance metric
"""
if __name__ == '__main__':
    model = SequentialRecommender(h=5, split_perc=0.0)
    #model.save_r_hat(evaluation=True)
    #model.test(distance=SequentialRecommender.SIM_COSINE, k=600,alpha=0.25,beta=0.5,shrink=10,l=0.25,c=0.5)
    sps.save_npz(model.get_r_hat())
