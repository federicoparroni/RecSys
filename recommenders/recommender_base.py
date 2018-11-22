from abc import abstractmethod
from abc import ABC
import utils.log as log
import numpy as np
import scipy.sparse as sps
import time
from utils.check_matrix_format import check_matrix
import os

class RecommenderBase(ABC):
    """ Defines the interface that all recommendations models expose """

    def __init__(self):
        self.name = 'recommenderbase'

    @abstractmethod
    def fit(self):
        """
        Fit the model on the data. Inherited class should extend this method in the appropriate way.
        """
        pass

    @abstractmethod
    def run(self):
        """
        Handle all the operations needed to run this model a single time.
        In particular, creates the object, performs the fit and get the recommendations.
        Then, it can either evaluate the recommendations or export the model
        """
        pass

    @abstractmethod
    def get_r_hat(self, load_from_file=False, path=''):
        """
        :param load_from_file: if the matrix has been saved can be set to true for load it from it
        :param path: path in which the matrix has been saved
        -------
        :return the extimated urm from the recommender, with just the target playlists rows
        """
        pass

    def save_r_hat(self):
        r_hat = self.get_r_hat()
        r_hat = check_matrix(r_hat, format='csr')

        # create dir if not exists
        filename = 'raw_data/saved_r_hat/{}_{}'.format(self.name, time.strftime('%H-%M-%S'))
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        sps.save_npz(filename, r_hat)


    @abstractmethod
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
    

    def recommend_batch(self, userids, urm,  N=10, filter_already_liked=True, with_scores=True, items_to_exclude=[], verbose=False):
        """
        Recommend the N best items for the specified list of users

        Parameters
        ----------
        userids : list of int
            The user ids to calculate recommendations for
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
            List of (user_id, recommendations), where recommendation
            is a list of length N of (itemid, score) tuples:
                [   [7,  [(18,0.7), (11,0.6), ...] ],
                    [13, [(65,0.9), (83,0.4), ...] ],
                    [25, [(30,0.8), (49,0.3), ...] ], ... ]
        """
        i=0
        L=len(userids)
        result = []
        for userid in userids:
            recs = self.recommend(userid, N=N, urm=urm, filter_already_liked=filter_already_liked,
                                    with_scores=with_scores, items_to_exclude=items_to_exclude)
            result.append(recs)
            if verbose:
                i+=1
                log.progressbar(i, L, prefix='Building recommendations ')
        return result

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

    def _insert_userids_as_first_col(self, userids, recommendations):
        """
        Add target id in a way that recommendations is a list as follows
        [ [playlist1_id, id1, id2, ....., id10], ...., [playlist_id2, id1, id2, ...] ]
        """
        np_target_id = np.array(userids)
        target_id_t = np.reshape(np_target_id, (len(np_target_id), 1))
        return np.concatenate((target_id_t, recommendations), axis=1)
