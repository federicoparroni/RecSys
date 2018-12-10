from recommenders.hybrid_base import Hybrid
from recommenders.hybrid_r_hat import HybridRHat
from recommenders.hybrid_similarity import HybridSimilarity
import scipy.sparse as sps
import numpy as np
from inout.importexport import exportcsv
from recommenders.hybrid.hybrid_cluster import HybridClusterInteractionsCount
from recommenders.content_based.content_based import ContentBasedRecommender
import data.data as data
import utils.cluster_ensemble as ce
from recommenders.collaborative_filtering.alternating_least_square import AlternatingLeastSquare
from recommenders.collaborative_filtering.pure_SVD import Pure_SVD
from recommenders.collaborative_filtering.userbased import CFUserBased
from recommenders.collaborative_filtering.SLIM_RMSE import SLIMElasticNetRecommender
from recommenders.collaborative_filtering.itembased import CFItemBased
from recommenders.content_based.content_based import ContentBasedRecommender
import os
import time
import clusterize.cluster as cluster
import recommenders.hybrid_base as hb
import utils.log as log


def weights_selection(models):
    WEIGHTS = []
    log.success('|SELECT THE WEIGHTS FOR THE MODELS|')
    for m in models:
        log.success('select the weights for :' + m)
        WEIGHTS.append(float(input()))
    return WEIGHTS

def normalization_mode_selection():
    log.success('|SELECT THE NORMALIZATION MODE|')
    log.warning('\'1\' MAX MATRIX')
    log.warning('\'2\' MAX ROW')
    log.warning('\'3\' L2 NORM')

    selection = input()[0]
    if selection == '1':
        NORMALIZATION_MODE = 'MAX_MATRIX'
    elif selection == '2':
        NORMALIZATION_MODE = 'MAX_ROW'
    elif selection == '3':
        NORMALIZATION_MODE = 'L2'
    else:
        log.info('wrong mode')
        exit(0)
    return NORMALIZATION_MODE

def option_selection(type):
    if type == 'SIM':
        # LET USER CHOOSE OPTIONS
        log.success('STUDY HARD | WORK HARD | FUCK HARD |')
        log.warning('\'s\' for save the r_hat in saved_r_hat')
        log.warning('\'e\' for save the r_hat in saved_r_hat_evaluation')
        option = input()[0]
        log.success('SELECT A NAME FOR THE MATRIX')
        name = input()

        if option == 's':
            urm_filter_tracks = data.get_urm()
            rel_path = 'saved_r_hat/'
        elif option == 'e':
            urm_filter_tracks = data.get_urm_train_1()
            rel_path = 'saved_r_hat_evaluation/'
        else:
            log.warning('CON UNA MANO SELEZIONI E CON L\'ALTRA FAI UNA SEGA AL TUO RAGAZZO...')
            exit(0)
        return name, urm_filter_tracks, rel_path
    elif type == 'R_HAT':
        # LET USER CHOOSE OPTIONS
        log.success('STUDY HARD | WORK HARD | FUCK HARD |')
        log.warning('\'s\' for save the r_hat in saved_r_hat')
        log.warning('\'e\' for EXPORT and get a SUB')
        option = input()[0]


        if option == 's':
            log.success('SELECT A NAME FOR THE MATRIX')
            name = input()
            urm_filter_tracks = data.get_urm()
            rel_path = 'saved_r_hat/'
            EXPORT = False
        elif option == 'e':
            log.success('SELECT A NAME FOR THE SUB')
            name = input()
            urm_filter_tracks = data.get_urm()
            rel_path = None
            EXPORT = True
        else:
            log.warning('CON UNA MANO SELEZIONI E CON L\'ALTRA FAI UNA SEGA AL TUO RAGAZZO...')
            exit(0)
        return name, urm_filter_tracks, rel_path, EXPORT

def LETS_FUCK_SOME_PUSSIES():

    SIM_MATRIX = ['saved_sim_matrix', 'saved_sim_matrix_evaluation']
    R_HAT = ['saved_r_hat', 'saved_r_hat_evaluation']
    SAVE = ['saved_sim_matrix', 'saved_r_hat']
    EVALUATE = ['saved_sim_matrix_evaluation', 'saved_r_hat_evaluation']

    log.error("""                              |~~~~~|        _____       _____
             _____                  \~~~/ /~~~~\ /   __|     /   __|
            |  =  |\                 | | |  o  / \  /  _  _  \  /
            |  =  | \           |~|  | | | /~~~   \ \  || ||  \ \ 
            |  =  |  |          \ \_/  / | |___    \ \ ||_||   \ \ 
     _______|  =  |__|____       \____/  |_____||\__| ||___||\__| |
    |          =          |\                    \____/      \____/
    | =================== | \ 
    |_______   =   _______|  |
     \      |  =  |\       \ |
      \_____|  =  | \_______\|
            |  =  |  |
            |  =  |  |
            |  =  |  |
            |  =  |  |
            |  =  |  |
            |  =  |  |
            |_____|  |  START TO PRAY... FOR A LITTLE PUSSY AND SOME GRAVES....
            \      \ | 
             \______\|""")

    start = time.time()

    matrices_array, folder, models = hb.create_matrices_array()

    print('matrices loaded in {:.2f} s'.format(time.time() - start))
    NORMALIZATION_MODE = normalization_mode_selection()

    if folder in SAVE:

        WEIGHTS = weights_selection(models)

        if folder in SIM_MATRIX:
            name, urm_filter_tracks, rel_path = option_selection('SIM')
            hybrid_rec = HybridSimilarity(matrices_array, normalization_mode=NORMALIZATION_MODE,
                                          urm_filter_tracks=urm_filter_tracks)
            sps.save_npz('raw_data/' + rel_path + name, hybrid_rec.get_r_hat(weights_array=WEIGHTS))
        if folder in R_HAT:
            name, urm_filter_tracks, rel_path, EXPORT = option_selection('R_HAT')
            hybrid_rec = HybridRHat(matrices_array, normalization_mode=NORMALIZATION_MODE,
                                    urm_filter_tracks=urm_filter_tracks)
            if EXPORT:
                recommendations = hybrid_rec.recommend_batch(weights_array=WEIGHTS,
                                                             target_userids=data.get_target_playlists())
                exportcsv(recommendations, path='submissions', name=name)
            else:
                sps.save_npz('raw_data/' + rel_path + name, hybrid_rec.get_r_hat(weights_array=WEIGHTS))

    elif folder in EVALUATE:

        log.success('|WHAT YOU WANT TO DO ???|')
        log.warning('\'1\' BAYESIAN SEARCH VALIDATION')
        log.warning('\'2\' HAND CRAFTED PARAMETERS')
        mode = input()[0]

        # BAYESIAN SEARCH
        if mode == '1':
            log.success('|SELECT A NUMBER OF |||ITERATIONS||| FOR THE ALGORITHM|')
            iterations = input()
            urm_filter_tracks = data.get_urm_train_1()
            if folder in SIM_MATRIX:
                hybrid_rec = HybridSimilarity(matrices_array, normalization_mode=NORMALIZATION_MODE,
                                              urm_filter_tracks=urm_filter_tracks)
            if folder in R_HAT:
                hybrid_rec = HybridRHat(matrices_array, normalization_mode=NORMALIZATION_MODE,
                                              urm_filter_tracks=urm_filter_tracks)
            hybrid_rec.validate(iterations=iterations, urm_test=data.get_urm_test_1(), userids=data.get_target_playlists())

        elif mode == '2':

            WEIGHTS = weights_selection(models)
            urm_filter_tracks = data.get_urm_train_1()
            if folder in SIM_MATRIX:
                hybrid_rec = HybridSimilarity(matrices_array, normalization_mode=NORMALIZATION_MODE,
                                              urm_filter_tracks=urm_filter_tracks)
            if folder in R_HAT:
                hybrid_rec = HybridRHat(matrices_array, normalization_mode=NORMALIZATION_MODE,
                                              urm_filter_tracks=urm_filter_tracks)
            recs = hybrid_rec.recommend_batch(weights_array=WEIGHTS, target_userids=data.get_target_playlists())
            hybrid_rec.evaluate(recommendations=recs, test_urm=data.get_urm_test_1())
    else:
        log.error('WRONG FOLDER')

if __name__ == '__main__':
    LETS_FUCK_SOME_PUSSIES()







