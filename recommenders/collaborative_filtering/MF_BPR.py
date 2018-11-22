from recommenders.recommender_base import RecommenderBase
import implicit
import data.data as data as d

class BPR(RecommenderBase):

    """
        A MF method with BPR loss.
        Actually working bad, it is just a draft
    """

    def fit(self, urm, n_factors=20, n_epochs=20, lr=0.007, reg=0.02, verbose=True):
        self.model = implicit.bpr.BayesianPersonalizedRanking(factors=n_factors, learning_rate=lr, regularization=reg,
                                                              iterations=n_epochs)
        item_user_data = urm.transpose()

        self.model.fit(item_user_data, show_progress=True)

    def recommend(self, userid, N=10, urm=None, filter_already_liked=True, with_scores=False, items_to_exclude=[]):
        if filter_already_liked:
            rec = self.model.recommend(userid=userid, user_items=urm, N=10)
        else:
            rec = self.model.recommend(userid=userid, N=10)

        if len(items_to_exclude) > 0:
            raise NotImplementedError('Items to exclude functionality is not implemented yet')

        if with_scores:
            return [userid] + [rec + [(-1, 0) for add_missing in range(10-len(rec))]]
        else:
            r,scores = zip(*rec)
            return [userid] + [j for j in r] + [-1 for add_missing in range(10-len(r))]

b = BPR()
b.fit(d.get_urm_train(), n_epochs=100, n_factors=20, lr=0.11, reg=0.2)
recs = b.recommend_batch(d.get_target_playlists(), N=10, urm=d.get_urm_train(), filter_already_liked=True,
                         with_scores=False, items_to_exclude=[])
b.evaluate(recs, d.get_urm_test(), print_result=True)
