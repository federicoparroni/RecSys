from abc import ABC, abstractmethod

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
        :param userid : int
            The user id to calculate recommendations for
        :param urm : csr_matrix
            A sparse matrix of shape (number_users, number_items). This allows to look
            up the liked items and their weights for the user. It is used to filter out
            items that have already been liked from the output, and to also potentially
            giving more information to choose the best items for this user.
        :param N : int, optional
            The number of recommendations to return
        :param items_to_exclude : list of ints, optional
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
        :param userids : list of int
            The user ids to calculate recommendations for
        :param urm : csr_matrix
            A sparse matrix of shape (number_users, number_items). This allows to look
            up the liked items and their weights for the user. It is used to filter out
            items that have already been liked from the output, and to also potentially
            giving more information to choose the best items for this user.
        :param N : int, optional
            The number of recommendations to return
        :param items_to_exclude : list of ints, optional
            List of extra item ids to filter out from the output

        Returns
        -------
        list
            List of (user_id, recommendations), where recommendation
            is a list of length N of (itemid, score) tuples:
                [   (7,  [(18,0.7), (11,0.6), ...] ),
                    (13, [(65,0.9), (83,0.4), ...] ),
                    (25, [(30,0.8), (49,0.3), ...] ), ... ]
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
                self._progressbar(i, L, prefix='Building recommendations ')
        return result

    def evaluate(self, recommendations, test_urm, at_k=10):
        """
        Return the MAP@k evaluation for the provided recommendations
        computed with respect to the test_urm

        Parameters
        ----------
        :param recommendations : list
            List of (user_id, recommendations), where recommendation
            is a list of length N of (itemid, score) tuples:
                [   (7,  [(18,0.7), (11,0.6), ...] ),
                    (13, [(65,0.9), (83,0.4), ...] ),
                    (25, [(30,0.8), (49,0.3), ...] ), ... ]
        :param test_urm : csr_matrix
            A sparse matrix
        :param at_k : int, optional
            The number of items to compute the precision at

        Returns
        -------
        :return (float) MAP@k: for the provided recommendations
        """
        assert at_k > 0

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

        return aps/len(recommendations)


    def _progressbar(self, iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
        """
        Call in a loop to print a progress bar

        Parameters
        ----------
        iteration:  int, current iteration
        total:      int, total iterations
        prefix:     str, prefix string
        suffix:     suffix string
        decimals:   int (optional), positive number of decimals in percent complete
        length:     int (optional), length of bar (in characters)
        fill:       int (optional), fill character
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('%s [%s] %s%% %s' % (prefix, bar, percent, suffix), end='\r')
        if iteration == total:  # print new line on complete
            print('')
