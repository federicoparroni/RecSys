from surprise import SVDpp
from recommenders.recommender_base import RecommenderBase
import numpy as np
from surprise import Reader
from surprise import data.data as dataset
import data.data as data as d
import pandas as pd

class SvdPP(RecommenderBase):

    """
        SVDpp algorithm.
        Actually woring bad, just a draft
    """

    def __init__(self, URM):

        print('train set built')
        # double check if training set is built fine for sgd
        # for u, i, r in self.trainset.all_ratings():
        #     a = 1

    def fit(self, urm, n_factors=20, n_epochs=20, lr_all=0.007, reg_all=0.02, init_mean=0,
            init_std_dev=0.1, verbose=True):
        # create the training set
        r, c = urm.nonzero()
        ones = np.ones(len(r), dtype=np.int32)
        d = np.vstack((r, c, ones)).transpose()
        df = pd.DataFrame(d)
        df.columns = ['userID', 'itemID', 'rating']
        reader = Reader()
        data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
        self.trainset = data.build_full_trainset()

        # fit
        self.algo = SVDpp(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, init_mean=init_mean,
                          init_std_dev=init_std_dev, verbose=verbose)
        self.algo.fit(self.trainset)

    def recommend(self, userid, N=10, urm=None, filter_already_liked=True, with_scores=True, items_to_exclude=[]):
        if len(items_to_exclude) > 1:
            raise NotImplementedError('Items to exclude functionality is not implemented yet')

        r = np.empty([1])
        for i in range(d.N_TRACKS):
            p = self.algo.predict(userid, i)
            r = np.array([p[3]]) if i == 0 else np.concatenate((r, np.array([p[3]])))

        if filter_already_liked:
            if urm == None:
                raise ValueError('Please provide a URM in order to items already liked')
            else:
                r[urm.getrow(userid).nonzero()[1]] = 0

        l = [userid]
        ind = np.argpartition(r, -10)[-10:]
        for i in ind:
            if with_scores:
                l.append((i, r[i]))
            else:
                l.append(i)
        return l


a = SvdPP(d.get_urm_train())
ids = d.get_target_playlists()[0: 49]
a.fit(n_epochs=5)
recs = a.recommend_batch(ids, N=10, urm=d.get_urm_train(), filter_already_liked=True, with_scores=False, items_to_exclude=[],
                         verbose=True)
a.evaluate(recs, d.get_urm_test())
