from abc import ABC, abstractmethod
import utils.log as log

class RecommenderBase(ABC):
    """ Defines the interface that all recommendations models expose """

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def recommend(self, userid, N=10, urm=None, filter_already_liked=True, with_scores=True, items_to_exclude=[]):
        """
        Recommends the N best items for the specified user

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
    
    def recommend_batch(self, userids, N=10, urm=None, filter_already_liked=True, with_scores=True, items_to_exclude=[], verbose=False):
        """
        Recommends the N best items for the specified list of users

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
            print('recommending {}'.format(userid))
            recs = self.recommend(userid, N=N, urm=urm, filter_already_liked=filter_already_liked,
                                    with_scores=with_scores, items_to_exclude=items_to_exclude)
            result.append(recs)
            if verbose:
                i+=1
                log.progressbar(i, L, prefix='Building recommendations ')
        return result

    def evaluate(self, recommendations, test_urm, at_k=10, print_result = True):
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
        :return (float) MAP@k: for the provided recommendations
        """
        if not at_k > 0:
            log.error('Invalid value of k {}'.format(at_k))
            return

        aps = 0
        for r in recommendations:
            row = test_urm.getrow(r[0]).indices
            m = min(at_k, len(row))

            ap = 0
            for j in range(1, m+1):
                n_elems_found = 0
                if r[j] in row:
                    n_elems_found += 1
                    ap += n_elems_found/j
            if m>0:
                ap /= m
                aps += ap

        result = aps/len(recommendations)
        if print_result:
            print('map: {}'.format(result))
        return result
