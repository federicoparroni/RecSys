
import utils.check_matrix_format as cm
from recommenders.Similarity_MFD.SimilarityMatrixRecommender import SimilarityMatrixRecommender
from recommenders.Similarity_MFD.Compute_Similarity import Compute_Similarity
import data.data as data
from recommenders.Similarity_MFD.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
import numpy as np
import utils.log as log
import scipy.sparse as sps
from utils.check_matrix_format import check_matrix
import time
import os

class UserKNNCFRecommender(SimilarityMatrixRecommender):
    """ UserKNN recommender"""

    RECOMMENDER_NAME = "UserKNNCFRecommender"

    def __init__(self, URM_train, sparse_weights=True):
        super(UserKNNCFRecommender, self).__init__()
        self.name = 'UserKNN'
        # Not sure if CSR here is faster
        self.URM_train = cm.check_matrix(URM_train, 'csr')

        self.dataset = None

        self.sparse_weights = sparse_weights

        self.compute_item_score = self.compute_score_user_based



    def fit(self, topK=150, shrink=10, similarity='tversky', normalize=True, **similarity_args):

        self.topK = topK
        self.shrink = shrink

        similarity = Compute_Similarity(self.URM_train.T, shrink=shrink, topK=topK, normalize=normalize,
                                        similarity=similarity, **similarity_args)

        if self.sparse_weights:
            self.W_sparse = similarity.compute_similarity()
        else:
            self.W = similarity.compute_similarity()
            self.W = self.W.toarray()

    def save_r_hat(self, evaluation):

        r_hat = self.W_sparse
        r_hat = check_matrix(r_hat, format='csr')

        # create dir if not exists
        if evaluation:
            filename = 'raw_data/saved_r_hat_evaluation/{}_{}'.format(self.name, time.strftime('%H-%M-%S'))
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        else:
            filename = 'raw_data/saved_r_hat/{}_{}'.format(self.name, time.strftime('%H-%M-%S'))
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        sps.save_npz(filename, r_hat)
        log.success('R_hat succesfully saved in: {}.npz'.format(filename))
        

if __name__ == '__main__':
    print()
    log.success('++ What do you want to do? ++')
    log.warning('(t) Test the model with some default params')
    log.warning('(r) Save the R^')
    log.warning('(s) Save the similarity matrix')
    #log.warning('(v) Validate the model')
    log.warning('(x) Exit')
    arg = input()[0]
    print()
    
    if arg == 't':
        # recs = model.recommend_batch(userids=data.get_target_playlists(), urm=data.get_urm_train())
        # model.evaluate(recommendations=recs, test_urm=data.get_urm_test())
        model = UserKNNCFRecommender(URM_train=data.get_urm_train_2())
        recs = model.recommend_batch(userids=data.get_target_playlists(), type='USER')
        model.evaluate(recs, test_urm=data.get_urm_test_2())
    elif arg == 'r':
        log.info('Wanna save for evaluation (y/n)?')
        choice = input()[0] == 'y'
        model = UserKNNCFRecommender(URM_train=data.get_urm_train_2())
        model.fit()
        print('Saving the R^...')
        model.save_r_hat(evaluation=choice)
    elif arg == 's':
        model = UserKNNCFRecommender(URM_train=data.get_urm_train_2())
        model.fit()
        print('Saving the similarity matrix...')
        sps.save_npz('raw_data/saved_sim_matrix_evaluation/{}'.format(model.name), model.W_sparse)
    # elif arg == 'v':
    #     model.validate(....)
    elif arg == 'x':
        pass
    else:
        log.error('Wrong option!')
