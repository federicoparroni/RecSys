from recommenders.recommender_base import RecommenderBase
import multiprocessing
from functools import partial
import pathos.pools as pp
import numpy as np
import scipy.sparse as sps
from sklearn.linear_model import ElasticNet
import time
import utils.log as log

completed = 0
old = 0
s_time = 0
r_time = 0

class SLIMElasticNetRecommender(RecommenderBase):
    """
    Train a Sparse Linear Methods (SLIM) item similarity model.
    NOTE: ElasticNet solver is parallel, a single intance of SLIM_ElasticNet will
          make use of half the cores available

    See:
        Efficient Top-N Recommendation by Linear Regression,
        M. Levy and K. Jack, LSRS workshop at RecSys 2013.
        https://www.slideshare.net/MarkLevy/efficient-slides

        SLIM: Sparse linear methods for top-n recommender systems,
        X. Ning and G. Karypis, ICDM 2011.
        http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf
    """

    def __init__(self, URM_train):
        self.URM_train = URM_train
        self.name = 'slim_rmse_elasticnet'

    def _partial_fit(self, URM_train, currentItem):

        if self.l1_ratio <= 0 or self.l1_ratio>1:
            print("SLIM_ElasticNet: l1_penalty+l2_penalty cannot be equal to zero, setting the ratio l1/(l1+l2) to 1.0")
            self.l1_ratio = 1.0

        # initialize the ElasticNet model
        model = ElasticNet(alpha=self.alpha,
                            l1_ratio=self.l1_ratio,
                            positive=self.positive_only,
                            fit_intercept=self.fit_intercept,
                            copy_X=self.copy_X,
                            precompute=self.precompute,
                            selection=self.selection,
                            max_iter=self.max_iter,
                            tol=self.tol)

        # get the target column
        y = URM_train[:, currentItem].toarray()

        # set the j-th column of X to zero
        start_pos = URM_train.indptr[currentItem]
        end_pos = URM_train.indptr[currentItem + 1]

        #current_item_data_backup = URM_train.data[start_pos: end_pos].copy()
        URM_train.data[start_pos: end_pos] = 0.0

        # fit one ElasticNet model per column
        model.fit(URM_train, y)

        # Select topK values
        # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
        # - Partition the data to extract the set of relevant items
        # - Sort only the relevant items
        # - Get the original item index
        if model.coef_.max() == 0:
            print('TORO IL MAX È 0 TROPPA REGOLARIZZAZIONE ;)')

        nonzero_model_coef_index = model.sparse_coef_.indices
        nonzero_model_coef_value = model.sparse_coef_.data

        local_topK = min(len(nonzero_model_coef_value) - 1, self.topK)

        relevant_items_partition = (-nonzero_model_coef_value).argpartition(local_topK)[0:local_topK]
        relevant_items_partition_sorting = np.argsort(-nonzero_model_coef_value[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]

        values = nonzero_model_coef_value[ranking]
        rows = nonzero_model_coef_index[ranking]

        #we need a column vector of same shape of the above rows vector filled with current item value
        cols = np.ones(rows.shape)
        cols = cols*currentItem

        global completed
        completed += 1

        global old
        global s_time
        global r_time

        #code for print
        if round(completed*100*4/20635, 2) > old:
            print(str(round(completed*100*4/20635, 2)) + '%')
            if time.clock()-r_time > 60:
                print(str(time.clock()-s_time) + 's elapsed from the start of the training')
                r_time = time.clock()
            old = round(completed*100*4/20635, 2)



        return values, rows, cols

    def fit(self,
            l1_ratio=0.1, positive_only=True, alpha=1e-4, fit_intercept=False, copy_X=False, precompute=False,
            selection='random', max_iter=100, topK=100, tol=1e-4, workers=multiprocessing.cpu_count()):

        """
        call this method for fit the model _pfit will be called from here

        fot these parameter see description from the ElasticNet class
        :param l1_ratio:
        :param positive_only:
        :param alpha:
        :param fit_intercept:
        :param copy_X:
        :param precompute:
        :param selection:
        :param max_iter:
        :param tol:

        :param topK: KNN, maximum number of elements of the W matrix columns (used fot remove noise)
        :param workers: number of parallel process, default set to the number of cpu core
        ---------
        :return: _
        """
        self.l1_ratio = l1_ratio
        self.positive_only = positive_only
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.precompute = precompute
        self.selection = selection
        self.max_iter = max_iter
        self.tol = tol

        self.topK = topK
        self.workers = workers

        self.URM_train = sps.csc_matrix(self.URM_train)
        n_items = self.URM_train.shape[1]

        #create a copy of the URM since each _pfit will modify it
        copy_urm = self.URM_train.copy()

        # oggetto riferito alla funzione nel quale predefinisco parte dell'input
        _pfit = partial(self._partial_fit, copy_urm)

        # creo un pool con un certo numero di processi
        pool = pp.ProcessPool(self.workers)

        # avvio il pool passando la funzione (con la parte fissa dell'input)
        # e il rimanente parametro, variabile
        print('train start')
        global s_time
        global r_time
        s_time = time.clock()
        r_time = time.clock()

        res = pool.map(_pfit, np.arange(n_items))

        # res contains a vector of (values, rows, cols) tuples
        values, rows, cols = [], [], []
        for values_, rows_, cols_ in res:
            values.extend(values_)
            rows.extend(rows_)
            cols.extend(cols_)

        # generate the sparse weight matrix
        self.W_sparse = sps.csc_matrix((values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)

    def recommend_batch(self, userids, N=10, filter_already_liked=True, with_scores=False, items_to_exclude=[], verbose=False):
        # compute the scores using the dot product
        user_profile = self.URM_train[userids]
        scores = user_profile.dot(self.W_sparse)

        if filter_already_liked:
            scores[user_profile.nonzero()] = -np.inf

        ranking = np.zeros((scores.shape[0], N), dtype=np.int)
        scores = scores.todense()

        for row_index in range(scores.shape[0]):
            scores_row = scores[row_index]

            relevant_items_partition = (-scores_row).argpartition(N)[0, 0:N]
            relevant_items_partition_sorting = np.argsort(-scores_row[0, relevant_items_partition])
            ranking[row_index] = relevant_items_partition[0, relevant_items_partition_sorting[0, 0:N]]

        """
        add target id in a way that recommendations is a list as follows
        [ [playlist1_id, id1, id2, ....., id10], ...., [playlist_id2, id1, id2, ...] ]
        """
        np_target_id = np.array(userids)
        target_id_t = np.reshape(np_target_id, (len(np_target_id), 1))
        recommendations = np.concatenate((target_id_t, ranking), axis=1)

        global completed
        completed=0

        return recommendations

    def recommend(self, userid, N=10, urm=None, filter_already_liked=True, with_scores=False, items_to_exclude=[]):
        pass

    def get_r_hat(self, load_from_file=False, path=''):
        """
        compute the r_hat for the model
        :return  r_hat
        """
        if load_from_file:
            r_hat = sps.load_npz(path)
        else:
            if self.W_sparse == None:
                log.error('the recommender has not been trained, call the fit() method for compute W')
            r_hat = self.URM_train.dot(self.W_sparse)
        return r_hat

