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
import utils.menu as menu

def symmetric_recommender_creator(models, type, normalization_mode, urm_filter_tracks):
    symmetric_matrices_array = []
    base_path = None
    if type == 'SIM':
        base_path = 'raw_data/saved_sim_matrix_evaluation_2/'
        for m in models:
            symmetric_matrices_array.append(sps.load_npz(base_path + m))
        return HybridSimilarity(symmetric_matrices_array, normalization_mode=normalization_mode, urm_filter_tracks=urm_filter_tracks)
    elif type == 'R_HAT':
        base_path = 'raw_data/saved_r_hat_evaluation_2/'
        for m in models:
            symmetric_matrices_array.append(sps.load_npz(base_path + m))
        return HybridRHat(symmetric_matrices_array, normalization_mode=normalization_mode, urm_filter_tracks=urm_filter_tracks)
    else:
        log.error('WRONG_TYPE')
        exit(0)

def weights_selection(models):
    WEIGHTS = []
    log.success('|SELECT THE WEIGHTS FOR THE MODELS|')
    for m in models:
        log.success('select the weights for: {}'.format(m))
        WEIGHTS.append(float(input()))
    return WEIGHTS

def ask_number_recommendations():
    log.success('Select the number of recommendations (default: 10)')
    N = int(input())
    return N

def normalization_mode_selection():
    log.success('|SELECT THE NORMALIZATION MODE|')
    log.warning('\'1\' MAX MATRIX')
    log.warning('\'2\' MAX ROW')
    log.warning('\'3\' L2 NORM')
    log.warning('\'4\' NONE')

    selection = input()[0]
    if selection == '1':
        NORMALIZATION_MODE = 'MAX_MATRIX'
    elif selection == '2':
        NORMALIZATION_MODE = 'MAX_ROW'
    elif selection == '3':
        NORMALIZATION_MODE = 'L2'
    elif selection == '4':
        NORMALIZATION_MODE = 'NONE'
    else:
        log.error('wrong mode')
        exit(0)
    return NORMALIZATION_MODE

def option_selection_save(type):
    if type == 'SIM':
        # LET USER CHOOSE OPTIONS
        log.success('STUDY HARD | WORK HARD | FUCK HARD |')
        log.warning('\'s\' for save the r_hat in saved_r_hat')
        #log.warning('\'e\' for save the r_hat in saved_r_hat_evaluation')
        option = input()[0]
        log.success('SELECT A NAME FOR THE MATRIX')
        name = input()

        if option == 's':
            urm_filter_tracks = data.get_urm()
            rel_path = 'saved_r_hat/'
        #elif option == 'e':
        #    urm_filter_tracks = data.get_urm_train_1()
        #    rel_path = 'saved_r_hat_evaluation/'
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

def option_selection_evaluation(type):
    if type == 'SIM':
        # LET USER CHOOSE OPTIONS
        log.success('STUDY HARD | WORK HARD | FUCK HARD |')
        log.warning('\'s\' for save the r_hat in saved_r_hat_evaluation')
        log.warning('\'m\' for compute the MAP@10')
        option = input()[0]

        if option == 's':
            urm_filter_tracks = data.get_urm_train_1()
            rel_path = 'saved_r_hat_evaluation/'
            log.success('SELECT A NAME FOR THE MATRIX')
            name = input()
        elif option == 'm':
            urm_filter_tracks = data.get_urm_train_1()
            rel_path = None
            name = None
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

def option_selection_evaluation_2():
    log.success('|EVALUATE OR SAVE THE MATRIX?|')
    log.warning('\'s\' save the matrix')
    log.warning('\'e\' evaluate the matrix')
    log.warning('\'c\' create the CSV')

    selection = input()[0]
    if selection in ['s', 'e', 'c']:
        return selection
    else:
        log.info('wrong mode')
        exit(0)

def export_csv_wizard(recommendations):
    log.info('Choose a name for the CSV:')
    name = input()
    exportcsv(recommendations, name=name)
    log.success('CSV saved!')


def wizard_hybrid():
    SIM_MATRIX = ['saved_sim_matrix', 'saved_sim_matrix_evaluation']
    R_HAT = ['saved_r_hat', 'saved_r_hat_evaluation']
    SAVE = ['saved_sim_matrix', 'saved_r_hat']
    EVALUATE = ['saved_sim_matrix_evaluation', 'saved_r_hat_evaluation']

    start = time.time()

    matrices_array, folder, models = hb.create_matrices_array()

    print('matrices loaded in {:.2f} s'.format(time.time() - start))
    log.success('You have loaded: {}'.format(models))

    NORMALIZATION_MODE = normalization_mode_selection()

    if folder in SAVE:
        WEIGHTS = weights_selection(models)

        if folder in SIM_MATRIX:
            name, urm_filter_tracks, rel_path = option_selection_save('SIM')
            hybrid_rec = HybridSimilarity(matrices_array, normalization_mode=NORMALIZATION_MODE,
                                          urm_filter_tracks=urm_filter_tracks)
            sps.save_npz('raw_data/' + rel_path + name, hybrid_rec.get_r_hat(weights_array=WEIGHTS))
        if folder in R_HAT:
            name, urm_filter_tracks, rel_path, EXPORT = option_selection_save('R_HAT')
            hybrid_rec = HybridRHat(matrices_array, normalization_mode=NORMALIZATION_MODE,
                                    urm_filter_tracks=urm_filter_tracks)
            if EXPORT:
                N = ask_number_recommendations()
                recommendations = hybrid_rec.recommend_batch(weights_array=WEIGHTS,
                                                             target_userids=data.get_target_playlists(), N=N)
                exportcsv(recommendations, path='submission', name=name)
            else:
                sps.save_npz('raw_data/' + rel_path + name, hybrid_rec.get_r_hat(weights_array=WEIGHTS))

    elif folder in EVALUATE:
        log.success('|WHAT YOU WANT TO DO ???|')
        log.warning('\'1\' BAYESIAN SEARCH VALIDATION')
        log.warning('\'2\' HAND CRAFTED WEIGHTS')
        mode = input()[0]

        # BAYESIAN SEARCH
        if mode == '1':
            log.success('|SELECT A NUMBER OF |||ITERATIONS||| FOR THE ALGORITHM|')
            iterations = float(input())
            urm_filter_tracks = data.get_urm_train_1()
            if folder in SIM_MATRIX:
                hybrid_rec = HybridSimilarity(matrices_array, normalization_mode=NORMALIZATION_MODE,
                                              urm_filter_tracks=urm_filter_tracks)
            if folder in R_HAT:
                hybrid_rec = HybridRHat(matrices_array, normalization_mode=NORMALIZATION_MODE,
                                              urm_filter_tracks=urm_filter_tracks)
            hybrid_rec.validate(iterations=iterations, urm_test=data.get_urm_test_1(), userids=data.get_target_playlists())

        # MANUAL WEIGHTS
        elif mode == '2':
            WEIGHTS = weights_selection(models)
            urm_filter_tracks = data.get_urm_train_1()
            chose = option_selection_evaluation_2()     # save, evaluate or csv
            if chose == 's':
                log.success('|CHOSE A NAME FOR THE MATRIX...|')
                name = input()
                if folder in SIM_MATRIX:
                    type = 'SIM'
                    hybrid_rec = HybridSimilarity(matrices_array, normalization_mode=NORMALIZATION_MODE,
                                                  urm_filter_tracks=urm_filter_tracks)
                elif folder in R_HAT:
                    type = 'R_HAT'
                    hybrid_rec = HybridRHat(matrices_array, normalization_mode=NORMALIZATION_MODE,
                                            urm_filter_tracks=urm_filter_tracks)

                sps.save_npz('raw_data/saved_r_hat_evaluation/' + name, hybrid_rec.get_r_hat(weights_array=WEIGHTS))
                sym_rec = symmetric_recommender_creator(models, type, NORMALIZATION_MODE,
                                                        urm_filter_tracks=data.get_urm_train_2())
                sps.save_npz('raw_data/saved_r_hat_evaluation_2/' + name, sym_rec.get_r_hat(weights_array=WEIGHTS))

            elif chose == 'e':
                if folder in SIM_MATRIX:
                    type = 'SIM'
                    hybrid_rec = HybridSimilarity(matrices_array, normalization_mode=NORMALIZATION_MODE,
                                                  urm_filter_tracks=urm_filter_tracks)
                elif folder in R_HAT:
                    type = 'R_HAT'
                    hybrid_rec = HybridRHat(matrices_array, normalization_mode=NORMALIZATION_MODE,
                                                  urm_filter_tracks=urm_filter_tracks)
                N = ask_number_recommendations()
                print('Recommending...')
                recs = hybrid_rec.recommend_batch(weights_array=WEIGHTS, target_userids=data.get_target_playlists(), N=N)
                hybrid_rec.evaluate(recommendations=recs, test_urm=data.get_urm_test_1())

                # export the recommendations
                log.success('Do you want to save the CSV with these recomendations? (y/n)')
                if input()[0] == 'y':
                    export_csv_wizard(recs)

                sym_rec = symmetric_recommender_creator(models, type, NORMALIZATION_MODE, urm_filter_tracks=data.get_urm_train_2())
                recs2 = sym_rec.recommend_batch(weights_array=WEIGHTS, target_userids=data.get_target_playlists())
                sym_rec.evaluate(recommendations=recs2, test_urm=data.get_urm_test_2())

            elif chose == 'c':
                if folder in R_HAT:
                    hybrid_rec = HybridRHat(matrices_array, normalization_mode=NORMALIZATION_MODE,
                                                  urm_filter_tracks=urm_filter_tracks)
                    N = ask_number_recommendations()
                    print('Recommending...')
                    recs = hybrid_rec.recommend_batch(weights_array=WEIGHTS, target_userids=data.get_target_playlists(), N=N)
                    
                    export_csv_wizard(recs)
                else:
                    log.error('not implemented yet')
    else:
        log.error('WRONG FOLDER')

def wizard_CF():
    print('Wizard soon...')

def wizard_CB():
    print('Wizard soon...')

def wizard_misc():
    print('Wizard soon...')

if __name__ == '__main__':
    print()
    log.error('(¯`·.¸¸.·´¯`·.¸¸.->  GabDamPar ® - The AIO Recommender System  <-.¸¸.·´¯`·.¸¸.·´¯)')
    print()

    menu.show('Which model do you want to run?', {
        '1': ('Collaborative Filtering', wizard_CF),
        '2': ('Content Based', wizard_CB),
        '3': ('Miscellaneous', wizard_misc),
        '4': ('Hybrid', wizard_hybrid),
    }, main_menu=True)

