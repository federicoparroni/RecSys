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
    log.info("""         _          __________                              _,
         _.-(_)._     ."          ".      .--""--.          _.-{__}-._
       .'________'.   | .--------. |    .'        '.      .:-'`____`'-:.
      [____________] /` |________| `\  /   .'``'.   \    /_.-"`_  _`"-._\ 
      /  / .\/. \  \|  / / .\/. \ \  ||  .'/.\/.\ '.  |  /`   / .\/. \   \ 
      |  \__/\__/  |\_/  \__/\__/  \_/|  : |_/\_| ;  |  |    \__/\__/    |
      \            /  \            /   \ '.\    /.' / .-\                /-.
      /'._  --  _.'\  /'._  --  _.'\   /'. `'--'` .'\/   '._-.__--__.-_.'   \ 
     /_   `""""`   _\/_   `""""`   _\ /_  `-./\.-'  _\ '.    `""""""""`    .'`\ 
    (__/    '|    \ _)_|           |_)_/            \__)|        '       |   |
      |_____'|_____|   \__________/   |              |;`_________'________`;-'
    jgs'----------'    '----------'   '--------------'`--------------------`
         S T A N          K Y L E        K E N N Y         C A R T M A N""")

    WEIGHTS = []
    log.success('|SELECT THE WEIGHTS FOR THE MODELS|')
    for m in models:
        log.success('select the weights for :' + m)
        WEIGHTS.append(float(input()))
    return WEIGHTS

def normalization_mode_selection():
    log.warning("""

  _____              __            .__  .__      __  .__                                        .__               
_/ ____\_ __   ____ |  | __ _____  |  | |  |   _/  |_|  |__   ____   ______  __ __  ______ _____|__| ____   ______
\   __\  |  \_/ ___\|  |/ / \__  \ |  | |  |   \   __\  |  \_/ __ \  \____ \|  |  \/  ___//  ___/  |/ __ \ /  ___/
 |  | |  |  /\  \___|    <   / __ \|  |_|  |__  |  | |   Y  \  ___/  |  |_> >  |  /\___ \ \___ \|  \  ___/ \___ \ 
 |__| |____/  \___  >__|_ \ (____  /____/____/  |__| |___|  /\___  > |   __/|____//____  >____  >__|\___  >____  >
                  \/     \/      \/                       \/     \/  |__|              \/     \/        \/     \/ 

""")
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
        log.info('wrong mode')
        exit(0)
    return NORMALIZATION_MODE

def option_selection_save(type):
    log.error("""

                                          .___                       .__                           .__                                      _____       .__                 
  ______ ____ ___  ___ _____    ____    __| _/   ____   ____    ____ |__| ____   ____   ___________|__| ____    ____     ______   ______  _/ ____\____  |  |   ______ ____  
 /  ___// __ \\  \/  / \__  \  /    \  / __ |  _/ __ \ /    \  / ___\|  |/    \_/ __ \_/ __ \_  __ \  |/    \  / ___\   /_____/  /_____/  \   __\\__  \ |  |  /  ___// __ \ 
 \___ \\  ___/ >    <   / __ \|   |  \/ /_/ |  \  ___/|   |  \/ /_/  >  |   |  \  ___/\  ___/|  | \/  |   |  \/ /_/  >  /_____/  /_____/   |  |   / __ \|  |__\___ \\  ___/ 
/____  >\___  >__/\_ \ (____  /___|  /\____ |   \___  >___|  /\___  /|__|___|  /\___  >\___  >__|  |__|___|  /\___  /                      |__|  (____  /____/____  >\___  >
     \/     \/      \/      \/     \/      \/       \/     \//_____/         \/     \/     \/              \//_____/                                  \/          \/     \/ 

""")
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
    log.error("""

                                              .___                       .__                           .__                                      _____       .__                 
      ______ ____ ___  ___ _____    ____    __| _/   ____   ____    ____ |__| ____   ____   ___________|__| ____    ____     ______   ______  _/ ____\____  |  |   ______ ____  
     /  ___// __ \\  \/  / \__  \  /    \  / __ |  _/ __ \ /    \  / ___\|  |/    \_/ __ \_/ __ \_  __ \  |/    \  / ___\   /_____/  /_____/  \   __\\__  \ |  |  /  ___// __ \ 
     \___ \\  ___/ >    <   / __ \|   |  \/ /_/ |  \  ___/|   |  \/ /_/  >  |   |  \  ___/\  ___/|  | \/  |   |  \/ /_/  >  /_____/  /_____/   |  |   / __ \|  |__\___ \\  ___/ 
    /____  >\___  >__/\_ \ (____  /___|  /\____ |   \___  >___|  /\___  /|__|___|  /\___  >\___  >__|  |__|___|  /\___  /                      |__|  (____  /____/____  >\___  >
         \/     \/      \/      \/     \/      \/       \/     \//_____/         \/     \/     \/              \//_____/                                  \/          \/     \/ 

    """)
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
    log.warning("""

      _____              __            .__  .__      __  .__                                        .__               
    _/ ____\_ __   ____ |  | __ _____  |  | |  |   _/  |_|  |__   ____   ______  __ __  ______ _____|__| ____   ______
    \   __\  |  \_/ ___\|  |/ / \__  \ |  | |  |   \   __\  |  \_/ __ \  \____ \|  |  \/  ___//  ___/  |/ __ \ /  ___/
     |  | |  |  /\  \___|    <   / __ \|  |_|  |__  |  | |   Y  \  ___/  |  |_> >  |  /\___ \ \___ \|  \  ___/ \___ \ 
     |__| |____/  \___  >__|_ \ (____  /____/____/  |__| |___|  /\___  > |   __/|____//____  >____  >__|\___  >____  >
                      \/     \/      \/                       \/     \/  |__|              \/     \/        \/     \/ 

    """)
    chose = ''
    log.success('|EVALUATE OR SAVE THE MATRIX?|')
    log.warning('\'s\' save the matrix')
    log.warning('\'e\' evaluate the matrix')

    selection = input()[0]
    if selection == 's':
        chose = 's'
    elif selection == 'e':
        chose = 'e'
    else:
        log.info('wrong mode')
        exit(0)
    return chose

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
            name, urm_filter_tracks, rel_path = option_selection_save('SIM')
            hybrid_rec = HybridSimilarity(matrices_array, normalization_mode=NORMALIZATION_MODE,
                                          urm_filter_tracks=urm_filter_tracks)
            sps.save_npz('raw_data/' + rel_path + name, hybrid_rec.get_r_hat(weights_array=WEIGHTS))
            log.info("""                  '#%n.
            ..++!%+:/X!!?:              .....
              "X!!!!!!!!!!?:       .+?!!!!!!!!!?h.
              X!!!!!!!!!!!!!k    n!!!!!!!!!!!!!!!!?:
             !!!!!!!!!!!!!!!!tMMMM!!!!!!!!!!!!!!!!!!?:
            X!!!!!!!!!!!!!!!!!5MMM``""%X!!!!!!!!!!!T4XL
            !!!!!!!!!X*")!!!!!!!?L       %X!!!!!!!!!k
           '!!!!!!X!`   !!XX!!!!!!L       'X!!!!!!!!!>
           'X!!!!M`    '" X!!!!!!!X        !!!!!!!!!!X
            X!!!f^~('>   X!!!!!!!!!        '!!!!!!!!!X
            X!!X     `.  X!!!!!!!!f        '!!!!!!!!!!
            !Xf  O    '  '!!!!!!X"         '!!!!!!!!!X
           "`f-.     :~  MX*t!X"           X!!!!!!!!!>
             >.         W! ~!             '!!!!!!!!!X
            :`             ':             X!!!!!!!!!!
            ~ ^`          '              '!!!!!!!!!X
            `~~~~~~~~     !              X!!!!!!!!!X
                `~~~      !             '!!!!!!!!!!>
                  >       !             (!!!!!!!!!X
                          !             !!!!!!!!!!X
                  >       '             '!!!!!!!!!!
                .:         /:`^-.        X!!!!!!!!!>
           /`  ~/         /(     `:       X!!!X!!!!!:
         /   : f         f'        !       `XX!M!!!!!%
        ~   / ~        :  >         4            "*!!!*
       ~   ~ '        ~   >          :
     :   :   !  >  .~     >          '
    /   /     ^~~"           .        ~
   ~   '                  `   \        !
 :     `                    ~-~"        `
        \                     > '(       '
!  ..:)./':                  ~   ':       ~
>.!!!!!!X   :              :       \~      >
\!!T!!!!!!                :       /
!?X$B!!!!!L ~                    ~        ~
?!!$$$X!!!X/             >     :        :
 `!!$$$!X 'N       f    ^-.   ~       .~
  '!!!!f` 'MMRn.    .-^"   ``       :`
    ~!X> ?RMMMMMMMR> .e*~        >`
      >`.XMMMMMMMMMMM~ ..    :~
      ds@MMMMMMMMMMMMNRMMMM5`
      RMMMMMMMMMMMMMMMMMMMMML
     'MMMMMMMMMMMMMMMMMMMMMMM
     9MMMMMMNMMMMMMMMMMMMMMMMK
     MMMMMMMMR8MMMMMMMMMMMMMMM
     RMMMMMMMMM$MMMMMMMMMMMMMMk
    'MMMMMMMMMMM$MMMMMMMMMMMMMM
    'MMMMMMMMMMMMMMMMMMMMMMMMMM""")
        if folder in R_HAT:
            name, urm_filter_tracks, rel_path, EXPORT = option_selection_save('R_HAT')
            hybrid_rec = HybridRHat(matrices_array, normalization_mode=NORMALIZATION_MODE,
                                    urm_filter_tracks=urm_filter_tracks)
            if EXPORT:
                recommendations = hybrid_rec.recommend_batch(weights_array=WEIGHTS,
                                                             target_userids=data.get_target_playlists())
                exportcsv(recommendations, path='submissions', name=name)
                log.success("""          7O
   @     GGG
  7R   CQGGQ
 SGR  GGGGGQ
7GGR #GGGGGQ      3GGGGGGS    SGC
RGGRRGGGGGGQ  7GGGGGGGGGGGG#(GGG(
QGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG
GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGQGGGO
GGGGGGGGGGGGGGGGGGGGGGQRGGGGGGGGGGG
GGGGGGGGGGGGGGGGGQRS///QGGGGGGGGGGR
GGGGGGGGGGGGGGQS(///^3QGGGGGGGGGGQ(
GGGGGGGGGGGQ3(//////SGGQS3GGGGGGGGQ
RGGGGGGGQ(//^///////RS(//RGGGGGGGGGt
   (((O /^^/^^^//^/////3RGGGGGGGGGGGC
     C///^/^^/^//^^^//^RGGGGGGGGGGGG7
    (7^//^/^^//((////^/QGGGGGGGGGGGQ(
    7R^ ^(S73/^   S//^^QGGRGGGGGGGGC
   O7     O3       (^/^GGQ3((QGGGGR
   S  OR(SO     /@  ///RR733S3QQ/
   #t  7S^O(       /^^/(////OO(
    7QCS^//(^     C3///////(/S
   O/^///////(OO33/////^/(7O#O
   C^^^^^/^/^^^^/^^/^^^^//(@RRR
    #(((/^^^^^^^^/^/^/^////RRBB@@C
   S(^/^^^//^^//^^////////^@RB@RRR@
  3^/^^/^//^^^^/^/^//^////(@RBBBB@BBO
 7/^^//^^//^/((OOSOO#(^///7@BBB@@@@BBC
   (3O3773SO//^/////^((///RBBB@B@@@BBR@t
         7RB@RO/^///^^////@BBB@@@@@@@BB@C
         BB@BBB#//^///////RB@B@@@@@@@B@BB#
         RB@BR@R@/^////^/RBB@@@@@@@@@B@@@BR
        ^B@@BB@@@Q///^//#B@@@@@@@@@@@@@@BBBO
        /@@@BB@@@7OOCO#@@@@@@@@@@@@@@@@@@BBR#
        7@@B@B@@BC    /RB@@@@@@@@@@@@B@@@BBBBB
""")
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
            iterations = float(input())
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
            chose = option_selection_evaluation_2()
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
                recs = hybrid_rec.recommend_batch(weights_array=WEIGHTS, target_userids=data.get_target_playlists())
                hybrid_rec.evaluate(recommendations=recs, test_urm=data.get_urm_test_1())

                sym_rec = symmetric_recommender_creator(models, type, NORMALIZATION_MODE, urm_filter_tracks=data.get_urm_train_2())
                recs2 = sym_rec.recommend_batch(weights_array=WEIGHTS, target_userids=data.get_target_playlists())
                sym_rec.evaluate(recommendations=recs2, test_urm=data.get_urm_test_2())

    else:
        log.error('WRONG FOLDER')

if __name__ == '__main__':
    LETS_FUCK_SOME_PUSSIES()






